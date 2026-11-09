//! Inference firmware: microphone window -> int8 CNN (nanoglia) -> zone id over USB serial.
//!
//! The model in `models/tap.tflite` is a placeholder (random weights) produced by
//! `host/scripts/export_tflite.py`; replace it with a trained one.
//! nanoglia loads the `.tflite` at run time: the weights are read in place from flash, and the
//! graph and its activations live in the static `MEMORY` arena (no heap).
#![no_std]
#![no_main]

use board::{Irqs, UsbDriver};
use embassy_executor::Spawner;
use embassy_nrf::pdm::{self, Frequency, OperationMode, Pdm, Ratio};
use embassy_time::{Instant, Timer};
use embassy_usb::UsbDevice;
use nanoglia::{Arena, Encoding, Graph, Target};
use panic_halt as _;

/// The model bytes, 16-byte aligned like the TFLite converter expects.
#[repr(C, align(16))]
struct Aligned<T: ?Sized>(T);

static MODEL: &Aligned<[u8]> = &Aligned(*include_bytes!("../models/tap.tflite"));

/// Memory for the graph and its execution context: the placeholder model needs ~4.6 KB
/// (graph 1.3 KB + context 3.3 KB). The line printed on connection gives the actual use.
static mut MEMORY: [u8; 8 * 1024] = [0; 8 * 1024];

/// Input window length in samples (must match the model input `(1, 1, WIN, 1)`).
const WIN: usize = 400;

#[embassy_executor::task]
async fn usb_task(mut device: UsbDevice<'static, UsbDriver>) -> ! {
    device.run().await
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = board::init();
    let _power = board::power_up_sensors(p.P0_22, p.P1_00, p.P0_17).await;

    let (device, mut serial) = board::usb_serial(p.USBD, "touchwood inference");
    spawner.spawn(usb_task(device).unwrap());

    // SAFETY: `main` runs once and is the only user of `MEMORY`.
    let arena = Arena::new(unsafe { &mut *core::ptr::addr_of_mut!(MEMORY) });
    let graph = Graph::load_in(&[&MODEL.0], Encoding::Autodetect, Target::Auto, &arena);
    let mut ctx = graph.as_ref().map_err(|e| *e).and_then(|g| g.init_execution_context_in(&arena));

    let config = pdm::Config {
        frequency: Frequency::_1280k, // 1.28 MHz / 80 = 16 kHz
        ratio: Ratio::Ratio80,
        operation_mode: OperationMode::Mono,
        ..Default::default()
    };
    let mut pdm = Pdm::new(p.PDM, Irqs, p.P0_26, p.P0_25, config);

    let mut window = [0i16; WIN];
    loop {
        serial.wait_connection().await;
        let ctx = match &mut ctx {
            Ok(ctx) => ctx,
            Err(error) => {
                // The model cannot run: keep saying why, so it shows up in any serial monitor.
                let line = Line::format(format_args!("model load failed: {error}\n"));
                let _ = board::write_all(&mut serial, line.as_bytes()).await;
                board::handle_bootloader_touch(&serial);
                Timer::after_secs(1).await;
                continue;
            }
        };
        let line = Line::format(format_args!(
            "nanoglia: arena {}/{} B\n",
            arena.used(),
            arena.capacity()
        ));
        if board::write_all(&mut serial, line.as_bytes()).await.is_err() {
            continue;
        }

        pdm.start().await;
        loop {
            board::handle_bootloader_touch(&serial);
            if pdm.sample(&mut window).await.is_err() {
                break;
            }

            let t0 = Instant::now();
            let result = (|| -> Result<(usize, f32), nanoglia::Error> {
                // Placeholder front-end: int16 -> int8 by dropping the low byte.
                // TODO: tap onset detection + the same normalisation as in training.
                // The input is rewritten before every compute: the memory plan reuses it.
                let mut input = ctx.input_mut(0)?;
                for (x, s) in input.as_mut_slice::<i8>()?.iter_mut().zip(&window) {
                    *x = (s >> 8) as i8;
                }
                ctx.compute()?;
                let output = ctx.output(0)?;
                let q = output.quantization();
                let scores = output.as_slice::<i8>()?;
                let zone = (0..scores.len()).max_by_key(|&i| scores[i]).unwrap_or(0);
                let confidence = q.scale() * (i32::from(scores[zone]) - q.zero_point()) as f32;
                Ok((zone, confidence))
            })();
            let dt = t0.elapsed().as_micros() as u32;

            let line = match result {
                Ok((zone, confidence)) => Line::format(format_args!(
                    "zone={zone} conf={:.0}% t={dt}us\n",
                    confidence * 100.0
                )),
                Err(error) => Line::format(format_args!("inference failed: {error}\n")),
            };
            if board::write_all(&mut serial, line.as_bytes()).await.is_err() {
                break;
            }
        }
        pdm.stop().await;
    }
}

/// One line of serial output, formatted without an allocator (truncated if too long).
struct Line {
    data: [u8; 96],
    len: usize,
}

impl Line {
    fn format(args: core::fmt::Arguments) -> Self {
        let mut line = Line { data: [0; 96], len: 0 };
        let _ = core::fmt::write(&mut line, args);
        line
    }

    fn as_bytes(&self) -> &[u8] {
        &self.data[..self.len]
    }
}

impl core::fmt::Write for Line {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let n = bytes.len().min(self.data.len() - self.len);
        self.data[self.len..self.len + n].copy_from_slice(&bytes[..n]);
        self.len += n;
        Ok(())
    }
}
