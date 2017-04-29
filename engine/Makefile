# What we're building
PRODUCT = mfcc

# What we're building with
CC = clang
CFLAGS = -Wall -m64
LDFLAGS = -arch x86_64

# Set debug flags and optimization level
ifeq ($(DEBUG), 1)
CFLAGS += -g -O0 -DDEBUG
else 
CFLAGS += -O3 -DNDEBUG
endif

# By default, make the product.
all:
	$(CC) $(CFLAGS) -shared -o $(PRODUCT).so $(PRODUCT).c

# How to clean up
clean:
	$(RM) $(PRODUCT).so
