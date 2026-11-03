//! Board support for the Arduino Nano 33 BLE Sense (nRF52840).
//!
//! The Arduino core does a few things silently at boot (see
//! `variants/ARDUINO_NANO33BLE/variant.cpp` in ArduinoCore-mbed). Without the
//! Arduino framework we have to do them ourselves:
//!
//! | What                       | Pin          | Note                                    |
//! |----------------------------|--------------|-----------------------------------------|
//! | Sensor rail (VDD_ENV)      | P0.22        | must be driven with *high drive*        |
//! | Internal I2C pull-ups      | P1.00        | enable ~10 ms after VDD_ENV             |
//! | Internal I2C (IMU, ...)    | SDA P0.14 / SCL P0.15 |                                |
//! | PDM microphone             | PWR P0.17, CLK P0.26, DIN P0.25 |                     |
//! | RGB LED (active low)       | R P0.24, G P0.16, B P0.06 |                            |
//! | Power LED                  | P1.09        |                                         |
//! | Bootloader entry           | `GPREGRET = 0xB0` + reset (Arduino "1200 baud touch") |
#![no_std]

use embassy_nrf::gpio::{Level, Output, OutputDrive};
use embassy_nrf::peripherals::{P0_17, P0_22, P1_00, TWISPI0, USBD};
use embassy_nrf::usb::vbus_detect::{self, HardwareVbusDetect};
use embassy_nrf::usb::{self, Driver};
use embassy_nrf::{bind_interrupts, pac, pdm, peripherals, twim, Peri};
use embassy_time::Timer;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::{Builder, UsbDevice};
use static_cell::StaticCell;

bind_interrupts!(pub struct Irqs {
    USBD => usb::InterruptHandler<peripherals::USBD>;
    CLOCK_POWER => vbus_detect::InterruptHandler;
    PDM => pdm::InterruptHandler<peripherals::PDM>;
    TWISPI0 => twim::InterruptHandler<peripherals::TWISPI0>;
});

/// Magic value understood by the Arduino bootloader (`DFU_MAGIC_SERIAL_ONLY_RESET`).
const BOOTLOADER_MAGIC: u8 = 0xB0;

/// Initialise the chip with the external 32 MHz crystal (required by USB).
pub fn init() -> embassy_nrf::Peripherals {
    let mut config = embassy_nrf::config::Config::default();
    config.hfclk_source = embassy_nrf::config::HfclkSource::ExternalXtal;
    embassy_nrf::init(config)
}

/// Keeps the power rails enabled for as long as it is alive.
pub struct SensorPower {
    _vdd_env: Output<'static>,
    _i2c_pullup: Output<'static>,
    _mic: Output<'static>,
}

/// Power the on-board sensors and the microphone, like the Arduino core does.
pub async fn power_up_sensors(
    vdd_env: Peri<'static, P0_22>,
    i2c_pullup: Peri<'static, P1_00>,
    mic_pwr: Peri<'static, P0_17>,
) -> SensorPower {
    let vdd_env = Output::new(vdd_env, Level::High, OutputDrive::HighDrive);
    Timer::after_millis(10).await;
    let i2c_pullup = Output::new(i2c_pullup, Level::High, OutputDrive::Standard);
    let mic = Output::new(mic_pwr, Level::High, OutputDrive::HighDrive);
    Timer::after_millis(10).await;
    SensorPower {
        _vdd_env: vdd_env,
        _i2c_pullup: i2c_pullup,
        _mic: mic,
    }
}

pub type UsbDriver = Driver<'static, HardwareVbusDetect>;
pub type Serial = CdcAcmClass<'static, UsbDriver>;

/// USB CDC-ACM serial port. Run the returned device in its own task.
pub fn usb_serial(usbd: Peri<'static, USBD>, product: &'static str) -> (UsbDevice<'static, UsbDriver>, Serial) {
    let driver = Driver::new(usbd, Irqs, HardwareVbusDetect::new(Irqs));

    // pid.codes test VID/PID: fine for development, not for distribution.
    let mut config = embassy_usb::Config::new(0x1209, 0x0001);
    config.manufacturer = Some("touchwood");
    config.product = Some(product);
    config.max_power = 100;
    config.max_packet_size_0 = 64;

    static CONFIG_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static BOS_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static MSOS_DESC: StaticCell<[u8; 256]> = StaticCell::new();
    static CONTROL_BUF: StaticCell<[u8; 64]> = StaticCell::new();
    static STATE: StaticCell<State> = StaticCell::new();

    let mut builder = Builder::new(
        driver,
        config,
        CONFIG_DESC.init([0; 256]),
        BOS_DESC.init([0; 256]),
        MSOS_DESC.init([0; 256]),
        CONTROL_BUF.init([0; 64]),
    );
    let class = CdcAcmClass::new(&mut builder, STATE.init(State::new()), 64);
    (builder.build(), class)
}

/// Write a buffer of any length as a sequence of 64-byte USB packets.
pub async fn write_all(serial: &mut Serial, data: &[u8]) -> Result<(), embassy_usb::driver::EndpointError> {
    for chunk in data.chunks(64) {
        serial.write_packet(chunk).await?;
    }
    Ok(())
}

/// Reboot into the Arduino bootloader if the host opened the port at 1200 baud,
/// so `bossac` can flash without a double tap on reset.
pub fn handle_bootloader_touch(serial: &Serial) {
    if serial.line_coding().data_rate() == 1200 {
        enter_bootloader();
    }
}

pub fn enter_bootloader() -> ! {
    cortex_m::interrupt::disable();
    pac::POWER.gpregret().write(|w| w.set_gpregret(BOOTLOADER_MAGIC));
    cortex_m::peripheral::SCB::sys_reset();
}

/// IMU fitted on the board, which tells the board revision.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Imu {
    /// Rev1: ST LSM9DS1 (accelerometer + gyroscope at I2C address 0x6B).
    Lsm9ds1,
    /// Rev2: Bosch BMI270 (accelerometer + gyroscope at I2C address 0x68).
    Bmi270,
    Unknown,
}

impl Imu {
    pub fn name(self) -> &'static str {
        match self {
            Imu::Lsm9ds1 => "LSM9DS1 (Rev1)",
            Imu::Bmi270 => "BMI270 (Rev2)",
            Imu::Unknown => "unknown",
        }
    }
}

/// Internal I2C bus (IMU, magnetometer, ...). Requires [`power_up_sensors`] first.
pub fn internal_i2c(
    twim: Peri<'static, TWISPI0>,
    sda: Peri<'static, peripherals::P0_14>,
    scl: Peri<'static, peripherals::P0_15>,
) -> twim::Twim<'static> {
    static TX_BUF: StaticCell<[u8; 32]> = StaticCell::new();
    let mut config = twim::Config::default();
    config.frequency = twim::Frequency::K400;
    twim::Twim::new(twim, Irqs, sda, scl, config, TX_BUF.init([0; 32]))
}

/// Identify the IMU through its WHO_AM_I / CHIP_ID register.
pub async fn detect_imu(i2c: &mut twim::Twim<'static>) -> Imu {
    let mut id = [0u8; 1];
    if i2c.write_read(0x6B, &[0x0F], &mut id).await.is_ok() && id[0] == 0x68 {
        return Imu::Lsm9ds1;
    }
    if i2c.write_read(0x68, &[0x00], &mut id).await.is_ok() && id[0] == 0x24 {
        return Imu::Bmi270;
    }
    Imu::Unknown
}
