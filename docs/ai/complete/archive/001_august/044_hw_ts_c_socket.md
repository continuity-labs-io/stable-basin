Context: We need to write the C boilerplate demonstrating how to extract raw
MAC/PHY hardware timestamps directly from a Network Interface Card (NIC),
bypassing the flawed OS software clock. This proves we understand the
foundational layer of IEEE 1588 PTP.

Task: Create a C program `src/modules/ptp_sync/02_hardware_timestamp.c` and a
corresponding `Makefile`.

1. Write a standard C network socket program designed to bind to a specific
   interface (e.g., `eth0`) and listen for incoming UDP packets on a designated
   port.
2. The critical step: Use `setsockopt` to enable the `SO_TIMESTAMPING` socket
   options. Specifically, enable `SOF_TIMESTAMPING_RX_HARDWARE` and
   `SOF_TIMESTAMPING_RAW_HARDWARE`.
3. Implement the `recvmsg` loop.
4. Extract the hardware timestamp from the packet's ancillary data (control
   messages) by parsing the `cmsghdr` structures looking for `SCM_TIMESTAMPING`.
5. Print the packet payload size and the extracted raw hardware timestamp
   (seconds and nanoseconds) to standard output.
6. Add detailed, highly technical comments explaining the memory structures, the
   difference between PHC (PTP Hardware Clock) and the system clock, and the
   `setsockopt` flags so an incoming C++ systems engineer knows we respect the
   bare-metal architecture.
