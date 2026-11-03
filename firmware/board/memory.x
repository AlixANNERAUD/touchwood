MEMORY
{
  /* The Arduino bootloader occupies the first 64 KiB; bossac writes the app at 0x10000. */
  FLASH : ORIGIN = 0x00010000, LENGTH = 1024K - 64K
  RAM   : ORIGIN = 0x20000000, LENGTH = 256K
}
