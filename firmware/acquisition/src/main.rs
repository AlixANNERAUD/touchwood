//! Data-collection firmware: streams the PDM microphone continuously over USB serial.
//!
//! Wire format (little-endian), one frame per message:
//!
//! | offset | size | field                                            |
//! |--------|------|--------------------------------------------------|
//! | 0      | 2    | magic `A5 5A`                                    |
//! | 2      | 1    | type: `0` = info (ASCII), `1` = audio (`i16` PCM) |
//! | 3      | 1    | reserved (0)                                     |
//! | 4      | 4    | sequence number (`u32`), audio blocks only       |
//! | 8      | 2    | payload length in bytes (`u16`)                  |
//! | 10     | n    | payload                                          |
//!
//! Audio blocks are numbered at capture time, so a gap in sequence numbers on the
//! host side means blocks were dropped (USB not keeping up, or port closed).
//!
//! TODO: stream the IMU once we know the board revision (it is only detected for now).
#![no_std]
#![no_main]

use core::sync::atomic::{AtomicU32, Ordering};

use board::{Irqs, Serial, UsbDriver};
use embassy_executor::Spawner;
use embassy_futures::join::join;
use embassy_nrf::pdm::{self, Frequency, OperationMode, Pdm, Ratio, SamplerState};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::channel::Channel;
use embassy_usb::UsbDevice;
use panic_halt as _;

/// Samples per audio block (16 ms at 16 kHz).
const BLOCK: usize = 256;
const SAMPLE_RATE: u32 = 16_000;
const HEADER: usize = 10;
const TYPE_INFO: u8 = 0;
const TYPE_AUDIO: u8 = 1;

/// Captured blocks waiting to be sent, tagged with their sequence number.
static AUDIO: Channel<CriticalSectionRawMutex, (u32, [i16; BLOCK]), 8> = Channel::new();
static DROPPED: AtomicU32 = AtomicU32::new(0);

#[embassy_executor::task]
async fn usb_task(mut device: UsbDevice<'static, UsbDriver>) -> ! {
    device.run().await
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = board::init();
    let _power = board::power_up_sensors(p.P0_22, p.P1_00, p.P0_17).await;

    let mut i2c = board::internal_i2c(p.TWISPI0, p.P0_14, p.P0_15);
    let imu = board::detect_imu(&mut i2c).await;

    let (device, mut serial) = board::usb_serial(p.USBD, "touchwood acquisition");
    spawner.spawn(usb_task(device).unwrap());

    let config = pdm::Config {
        frequency: Frequency::_1280k, // 1.28 MHz / 80 = 16 kHz
        ratio: Ratio::Ratio80,
        operation_mode: OperationMode::Mono,
        ..Default::default()
    };
    let mut pdm = Pdm::new(p.PDM, Irqs, p.P0_26, p.P0_25, config);

    // Capture: runs forever, never blocks (drops blocks if the sender falls behind).
    let capture = async {
        let mut bufs = [[0i16; BLOCK]; 2];
        let mut seq: u32 = 0;
        let _ = pdm
            .run_task_sampler(&mut bufs, |buf| {
                if AUDIO.try_send((seq, *buf)).is_err() {
                    DROPPED.fetch_add(1, Ordering::Relaxed);
                }
                seq = seq.wrapping_add(1);
                SamplerState::Sampled
            })
            .await;
    };

    // Send: one session per host connection.
    let send = async {
        loop {
            serial.wait_connection().await;
            AUDIO.clear();
            if send_info(&mut serial, imu.name()).await.is_err() {
                continue;
            }
            while stream_one(&mut serial).await.is_ok() {}
        }
    };

    join(capture, send).await;
}

async fn send_info(serial: &mut Serial, imu: &str) -> Result<(), embassy_usb::driver::EndpointError> {
    let mut text = Buf::<128>::new();
    let _ = core::fmt::write(
        &mut text,
        format_args!("touchwood acquisition v0; fs={SAMPLE_RATE}; block={BLOCK}; imu={imu}"),
    );
    let mut frame = [0u8; HEADER + 128];
    let n = encode_header(&mut frame, TYPE_INFO, 0, text.len);
    frame[n..n + text.len].copy_from_slice(&text.data[..text.len]);
    board::write_all(serial, &frame[..n + text.len]).await
}

async fn stream_one(serial: &mut Serial) -> Result<(), embassy_usb::driver::EndpointError> {
    let (seq, block) = AUDIO.receive().await;
    board::handle_bootloader_touch(serial);
    let mut frame = [0u8; HEADER + 2 * BLOCK];
    let n = encode_header(&mut frame, TYPE_AUDIO, seq, 2 * BLOCK);
    for (i, s) in block.iter().enumerate() {
        frame[n + 2 * i..n + 2 * i + 2].copy_from_slice(&s.to_le_bytes());
    }
    board::write_all(serial, &frame).await
}

fn encode_header(frame: &mut [u8], kind: u8, seq: u32, len: usize) -> usize {
    frame[0] = 0xA5;
    frame[1] = 0x5A;
    frame[2] = kind;
    frame[3] = 0;
    frame[4..8].copy_from_slice(&seq.to_le_bytes());
    frame[8..10].copy_from_slice(&(len as u16).to_le_bytes());
    HEADER
}

/// Tiny fixed-capacity `fmt::Write` buffer (no allocator on the MCU).
struct Buf<const N: usize> {
    data: [u8; N],
    len: usize,
}

impl<const N: usize> Buf<N> {
    fn new() -> Self {
        Self { data: [0; N], len: 0 }
    }
}

impl<const N: usize> core::fmt::Write for Buf<N> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let n = bytes.len().min(N - self.len);
        self.data[self.len..self.len + n].copy_from_slice(&bytes[..n]);
        self.len += n;
        Ok(())
    }
}
