CC = gcc
CFLAGS = -Wall -Wextra -O2
TARGET = 02_hardware_timestamp.bin
SRC = 02_hardware_timestamp.c

# OS Detection
UNAME_S := $(shell uname -s)

all: $(TARGET)

$(TARGET): $(SRC)
ifeq ($(UNAME_S),Linux)
	$(CC) $(CFLAGS) -o $@ $^
else
	@echo "Warning: Compiling on non-Linux OS ($(UNAME_S))."
	@echo "The boilerplate will compile, but SO_TIMESTAMPING and PHC features are Linux-specific."
	$(CC) $(CFLAGS) -o $@ $^
endif

clean:
	rm -f $(TARGET)
