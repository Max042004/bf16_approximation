CC := gcc
CFLAGS := -Wall -Wextra -lm -O2 -std=c11
LDLIBS := -lm
TARGET := bf16_test

SRCS := bf16.c bf16_test.c
OBJS := $(SRCS:.c=.o)
HDRS := bf16.h

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@ 

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
