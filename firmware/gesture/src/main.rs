//! TP 5: deploy the TP 2 gesture classifier on the Nano 33 BLE Sense, with nanoglia instead of
//! MicroTFLite.
//!
//! On each serial connection (115200 bauds or any rate, it is USB):
//! 1. model information, like MicroTFLite's `ModelPrint*` functions (section 3.1);
//! 2. the fixed test windows of `test_windows.rs`, one line per window with the probabilities and
//!    the latency, to compare with the reference ones printed by the host script (section 3.2);
//! 3. real-time classification from the LSM9DS1, one prediction per WINDOW rows (section 4).
//!
//! `models/gesture_model.tflite` and `src/test_windows.rs` (with WINDOW and the class order) come
//! from `host/scripts/tp5_gesture.py`.
//!
//! The TP 2 recordings are not what the TP describes (accelerometer in g at 100 Hz): they are the
//! gyroscope in dps (steps of 2000/32768), logged at ~422 rows/s from a sensor running at 119 Hz,
//! each value repeated 3 or 4 times. Real time reproduces that logger: the gyroscope output is read
//! every 2.37 ms whether or not it holds a new measurement.
#![no_std]
#![no_main]

mod test_windows;

use board::{Imu, Line, Serial, UsbDriver};
use embassy_executor::Spawner;
use embassy_nrf::twim::Twim;
use embassy_time::{Duration, Instant, Ticker};
use embassy_usb::driver::EndpointError;
use embassy_usb::UsbDevice;
use nanoglia::{Arena, DataType, Encoding, Graph, Target, TensorDesc};
use panic_halt as _;
use test_windows::{CLASSES, N_TEST, TEST_LABELS, TEST_WINDOWS, WINDOW};

const N_AXES: usize = 3;
/// Size of the flattened input, interleaved: x0, y0, z0, x1, y1, z1, ...
const N_INPUTS: usize = WINDOW * N_AXES;
const N_CLASSES: usize = CLASSES.len();
// The lines printed below show four probabilities.
const _: () = assert!(N_CLASSES == 4);
/// Row period of the TP 2 recordings: 119 Hz x 3.55 rows per measurement = ~422 rows/s.
const SAMPLE_PERIOD: Duration = Duration::from_micros(2_370);

/// The model bytes, 16-byte aligned like the TFLite converter expects (the `alignas(16)` of
/// `gesture_model.h`).
#[repr(C, align(16))]
struct Aligned<T: ?Sized>(T);

static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!("../models/gesture_model.tflite"));

/// Working memory of the interpreter (the TP's `tensorArena`): the TP 2 model uses 16.8 KB (graph
/// 2.3 KB + context 14.4 KB, measured with nanoglia on the PC), so used + ~1 KB as in section 3.1.
/// The connection message gives the bytes used: adjust if another model gets close.
static mut MEMORY: [u8; 18 * 1024] = [0; 18 * 1024];

/// Format and send one line; `?` on it ends the session when the host closes the port.
macro_rules! say {
    ($serial:expr, $($arg:tt)*) => {
        board::write_all($serial, Line::format(format_args!($($arg)*)).as_bytes()).await
    };
}

#[embassy_executor::task]
async fn usb_task(mut device: UsbDevice<'static, UsbDriver>) -> ! {
    device.run().await
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = board::init();
    let _power = board::power_up_sensors(p.P0_22, p.P1_00, p.P0_17).await;
    let (device, mut serial) = board::usb_serial(p.USBD, "touchwood gesture");
    spawner.spawn(usb_task(device).unwrap());

    let mut i2c = board::internal_i2c(p.TWISPI0, p.P0_14, p.P0_15);
    let imu = board::detect_imu(&mut i2c).await;
    let imu_ok = imu == Imu::Lsm9ds1 && lsm9ds1::init(&mut i2c).await.is_ok();

    // SAFETY: `main` runs once and is the only user of `MEMORY`.
    let arena = Arena::new(unsafe { &mut *core::ptr::addr_of_mut!(MEMORY) });
    let graph = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena);
    let mut ctx = graph
        .as_ref()
        .map_err(|e| *e)
        .and_then(|g| g.init_execution_context_in(&arena));

    loop {
        serial.wait_connection().await;
        board::handle_bootloader_touch(&serial);
        let (graph, ctx) = match (&graph, &mut ctx) {
            (Ok(graph), Ok(ctx)) => (graph, ctx),
            (_, Err(error)) => {
                // Also set when the graph failed: the context error is the graph's.
                let _ = say!(&mut serial, "Error: model load failed: {error}\r\n");
                embassy_time::Timer::after_secs(1).await;
                continue;
            }
            (Err(_), Ok(_)) => unreachable!(),
        };
        // Ends with Err when the host closes the port: wait for the next connection.
        let _ = session(&mut serial, &mut i2c, imu, imu_ok, &arena, graph, ctx).await;
    }
}

async fn session<A: nanoglia::Allocator, B: nanoglia::Allocator>(
    serial: &mut Serial,
    i2c: &mut Twim<'static>,
    imu: Imu,
    imu_ok: bool,
    arena: &Arena<'_>,
    graph: &Graph<'_, A>,
    ctx: &mut nanoglia::ExecutionContext<'_, '_, A, B>,
) -> Result<(), EndpointError> {
    // 3.1: model information.
    say!(serial, "\r\n== TP 5: gesture classifier (nanoglia) ==\r\n")?;
    say!(serial, "IMU: {}\r\n", imu.name())?;
    say!(serial, "Operators: {}\r\n", graph.operator_count())?;
    for i in 0..graph.operator_count() {
        say!(serial, "  {i}: {}\r\n", graph.kernel_name(i).unwrap_or("?"))?;
    }
    if let (Ok(input), Ok(output)) = (graph.input(0), graph.output(0)) {
        print_tensor(serial, "Input", &input).await?;
        print_tensor(serial, "Output", &output).await?;
        let n_in: i32 = input.dims.as_slice().iter().product();
        let n_out: i32 = output.dims.as_slice().iter().product();
        if n_in as usize != N_INPUTS || n_out as usize != N_CLASSES {
            say!(
                serial,
                "Error: the model expects {n_in} inputs / {n_out} outputs, \
                          the firmware {N_INPUTS} / {N_CLASSES}: fix WINDOW or N_CLASSES\r\n"
            )?;
            return wait_forever(serial).await;
        }
    }
    say!(
        serial,
        "Arena used: {} B of {} B (graph {} B, context {} B)\r\n",
        arena.used(),
        arena.capacity(),
        graph.graph_bytes(),
        graph.context_bytes()
    )?;

    // 3.2: fixed test windows.
    say!(serial, "\r\n-- Test windows --\r\n")?;
    for k in 0..N_TEST {
        let t0 = Instant::now();
        let result = classify(ctx, &TEST_WINDOWS[k]);
        let latency_us = t0.elapsed().as_micros();
        match result {
            Ok((pred, p)) => say!(
                serial,
                "Window {k} | expected: {} | predicted: {} | probs: {:.3} {:.3} {:.3} {:.3} | latency: {latency_us} us\r\n",
                CLASSES[TEST_LABELS[k]],
                CLASSES[pred],
                p[0], p[1], p[2], p[3]
            )?,
            Err(error) => say!(serial, "Window {k}: inference failed: {error}\r\n")?,
        }
    }

    // 4: real time.
    if !imu_ok {
        say!(
            serial,
            "\r\nNo LSM9DS1 (found: {}): no real-time classification\r\n",
            imu.name()
        )?;
        return wait_forever(serial).await;
    }
    say!(
        serial,
        "\r\n-- Real time (gyroscope, {WINDOW} rows of 2.37 ms per window) --\r\n"
    )?;
    let mut buffer = [[0f32; N_AXES]; WINDOW];
    let mut n_samples = 0;
    let mut flat = [0f32; N_INPUTS];
    let mut ticker = Ticker::every(SAMPLE_PERIOD);
    loop {
        ticker.next().await;
        board::handle_bootloader_touch(serial);
        // Like the TP 2 logger: one row per tick, repeating the last measurement until the next one.
        let Ok(rate) = lsm9ds1::read_gyroscope(i2c).await else {
            continue;
        };
        buffer[n_samples] = rate;
        n_samples += 1;
        if n_samples == WINDOW {
            n_samples = 0;
            // Raw values in dps, no normalisation, flattened interleaved like at training.
            for (i, sample) in buffer.iter().enumerate() {
                flat[i * N_AXES..(i + 1) * N_AXES].copy_from_slice(sample);
            }
            let t0 = Instant::now();
            match classify(ctx, &flat) {
                Ok((pred, p)) => say!(
                    serial,
                    "{:8} {:.3} | {:.3} {:.3} {:.3} {:.3} | {} us\r\n",
                    CLASSES[pred],
                    p[pred],
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    t0.elapsed().as_micros()
                )?,
                Err(error) => say!(serial, "inference failed: {error}\r\n")?,
            }
        }
    }
}

/// Like `ModelSetInput` + `ModelRunInference` + `ModelGetOutput`: float values in, quantized with
/// the input scale and zero point, probabilities dequantized out. Returns (argmax, probabilities).
fn classify<A: nanoglia::Allocator, B: nanoglia::Allocator>(
    ctx: &mut nanoglia::ExecutionContext<'_, '_, A, B>,
    values: &[f32],
) -> nanoglia::Result<(usize, [f32; N_CLASSES])> {
    // The input is rewritten before every compute: the memory plan reuses it.
    let mut input = ctx.input_mut(0)?;
    if input.dtype() == DataType::F32 {
        input.as_mut_slice::<f32>()?.copy_from_slice(values);
    } else {
        let q = input.quantization();
        let (scale, zero_point) = (q.scale(), q.zero_point() as f32);
        for (x, v) in input.as_mut_slice::<i8>()?.iter_mut().zip(values) {
            *x = round_half_even(v / scale + zero_point).clamp(-128.0, 127.0) as i8;
        }
    }
    ctx.compute()?;

    let output = ctx.output(0)?;
    let mut probs = [0f32; N_CLASSES];
    if output.dtype() == DataType::F32 {
        probs.copy_from_slice(output.as_slice::<f32>()?);
    } else {
        let q = output.quantization();
        for (p, &x) in probs.iter_mut().zip(output.as_slice::<i8>()?) {
            *p = q.scale() * (i32::from(x) - q.zero_point()) as f32;
        }
    }
    let mut pred = 0;
    for j in 1..N_CLASSES {
        if probs[j] > probs[pred] {
            pred = j;
        }
    }
    Ok((pred, probs))
}

/// Round half to even, like `np.round` in the reference code, so the quantized inputs are the
/// same as on the PC (`f32::round` needs `std`, and rounds half away from zero anyway).
fn round_half_even(x: f32) -> f32 {
    let floor = {
        let t = x as i32 as f32; // truncation toward zero
        if t > x {
            t - 1.0
        } else {
            t
        }
    };
    let diff = x - floor;
    if diff > 0.5 || (diff == 0.5 && floor as i32 % 2 != 0) {
        floor + 1.0
    } else {
        floor
    }
}

async fn print_tensor(serial: &mut Serial, what: &str, t: &TensorDesc<'_>) -> Result<(), EndpointError> {
    say!(
        serial,
        "{what}: dims {:?} {:?}, scale {} zero point {}\r\n",
        t.dims,
        t.dtype,
        t.quant.scale(),
        t.quant.zero_point()
    )
}

/// Keep the session open (a message stays on screen) until the host closes the port.
async fn wait_forever(serial: &mut Serial) -> Result<(), EndpointError> {
    loop {
        embassy_time::Timer::after_secs(1).await;
        board::handle_bootloader_touch(serial);
        if !serial.dtr() {
            return Err(EndpointError::Disabled);
        }
    }
}

/// The LSM9DS1 gyroscope, configured like the `Arduino_LSM9DS1` library.
mod lsm9ds1 {
    use embassy_nrf::twim::{Error, Twim};

    const ADDRESS: u8 = 0x6B;
    const CTRL_REG1_G: u8 = 0x10;
    const OUT_X_G: u8 = 0x18;

    /// 119 Hz, +/-2000 dps, as `IMU.begin()`.
    pub async fn init(i2c: &mut Twim<'static>) -> Result<(), Error> {
        i2c.write(ADDRESS, &[CTRL_REG1_G, 0x78]).await
    }

    /// `IMU.readGyroscope(x, y, z)`: the latest measurement in dps, new or not.
    pub async fn read_gyroscope(i2c: &mut Twim<'static>) -> Result<[f32; 3], Error> {
        let mut raw = [0u8; 6];
        i2c.write_read(ADDRESS, &[OUT_X_G], &mut raw).await?;
        let dps = |i: usize| f32::from(i16::from_le_bytes([raw[i], raw[i + 1]])) * 2000.0 / 32768.0;
        Ok([dps(0), dps(2), dps(4)])
    }
}
