# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wextra
TARGET = mfea
SOURCE = TRP__TSP.cpp

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Run with example input
test: $(TARGET)
	./$(TARGET) < input_example.txt

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

# Release build with optimizations
release: CXXFLAGS += -O3 -DNDEBUG
release: clean $(TARGET)

# Install (copy to system path)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# Uninstall
uninstall:
	rm -f /usr/local/bin/$(TARGET)

# Format code (requires clang-format)
format:
	clang-format -i $(SOURCE)

# Static analysis (requires cppcheck)
check:
	cppcheck --enable=all --std=c++11 $(SOURCE)

# Create results directory
setup:
	mkdir -p results

.PHONY: all clean test debug release install uninstall format check setup
