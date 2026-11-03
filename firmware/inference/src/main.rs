//! Inference firmware: microphone window -> int8 CNN (MicroFlow) -> zone id over USB serial.
//!
//! The model in `models/tap.tflite` is a placeholder (random weights) produced by
//! `host/scripts/export_microflow.py`; replace it with a trained one.
//! MicroFlow compiles the `.tflite` into Rust at build time: weights live in flash,
//! activations on the stack.
#![no_std]
#![no_main]

use board::{Irqs, UsbDriver};
use embassy_executor::Spawner;
use embassy_nrf::pdm::{self, Frequency, OperationMode, Pdm, Ratio};
use embassy_time::Instant;
use embassy_usb::UsbDevice;
use microflow::buffer::Buffer4D;
use microflow::model;
use panic_halt as _;

// Path is relative to the workspace root (MicroFlow reads it from rustc's working directory).
#[model("inference/models/tap.tflite")]
struct TapModel;

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
        pdm.start().await;
        loop {
            board::handle_bootloader_touch(&serial);
            if pdm.sample(&mut window).await.is_err() {
                break;
            }
            // Placeholder front-end: int16 -> int8 by dropping the low byte.
            // TODO: tap onset detection + the same normalisation as in training.
            let input: Buffer4D<i8, 1, 1, WIN, 1> =
                [nalgebra::SMatrix::from_fn(|_, j| [(window[j] >> 8) as i8])];

            let t0 = Instant::now();
            let probs = TapModel::predict_quantized(input);
            let dt = t0.elapsed().as_micros() as u32;

            let (_, zone) = probs.iamax_full(); // (row, col) of the highest probability
            let line = Line::format(zone as u8, probs[(0, zone)] * 100.0, dt);
            if board::write_all(&mut serial, line.as_bytes()).await.is_err() {
                break;
            }
        }
        pdm.stop().await;
    }
}

/// `zone=<z> conf=<pct>% t=<us>us\n`, formatted without an allocator.
struct Line {
    data: [u8; 48],
    len: usize,
}

impl Line {
    fn format(zone: u8, confidence_pct: f32, latency_us: u32) -> Self {
        let mut line = Line { data: [0; 48], len: 0 };
        let _ = core::fmt::write(
            &mut line,
            format_args!("zone={zone} conf={:.0}% t={latency_us}us\n", confidence_pct),
        );
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
