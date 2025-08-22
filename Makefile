# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wextra
TARGET = mfea
HYBRID_TARGET = mfea_gls_fls
SOURCE = TRP__TSP.cpp
HYBRID_SOURCE = GLS/TSP_TRPGLS.cpp

# Default target - build both versions
all: $(TARGET) $(HYBRID_TARGET)

# Build standard MFEA
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

# Build hybrid MFEA+GLS+FLS
$(HYBRID_TARGET): $(HYBRID_SOURCE)
	$(CXX) $(CXXFLAGS) $(HYBRID_SOURCE) -o $(HYBRID_TARGET)

# Alias for hybrid version
hybrid: $(HYBRID_TARGET)
gls: $(HYBRID_TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(HYBRID_TARGET)

# Run standard version with example input
test: $(TARGET)
	./$(TARGET) < input_example.txt

# Run hybrid version with example input  
test-hybrid: $(HYBRID_TARGET)
	./$(HYBRID_TARGET) < input_example.txt

# Debug builds
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

debug-hybrid: CXXFLAGS += -g -DDEBUG  
debug-hybrid: $(HYBRID_TARGET)

# Release builds with optimizations
release: CXXFLAGS += -O3 -DNDEBUG
release: clean $(TARGET)

release-hybrid: CXXFLAGS += -O3 -DNDEBUG
release-hybrid: clean $(HYBRID_TARGET)

# Install both versions
install: $(TARGET) $(HYBRID_TARGET)
	cp $(TARGET) /usr/local/bin/
	cp $(HYBRID_TARGET) /usr/local/bin/

# Uninstall
uninstall:
	rm -f /usr/local/bin/$(TARGET)
	rm -f /usr/local/bin/$(HYBRID_TARGET)

# Format code (requires clang-format)
format:
	clang-format -i $(SOURCE) $(HYBRID_SOURCE)

# Static analysis (requires cppcheck)
check:
	cppcheck --enable=all --std=c++11 $(SOURCE)
	cppcheck --enable=all --std=c++11 $(HYBRID_SOURCE)

# Create results directory
setup:
	mkdir -p results

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build both standard and hybrid versions"
	@echo "  mfea         - Build standard MFEA"
	@echo "  hybrid/gls   - Build hybrid MFEA+GLS+FLS" 
	@echo "  test         - Test standard version"
	@echo "  test-hybrid  - Test hybrid version"
	@echo "  clean        - Remove build artifacts"
	@echo "  debug        - Debug build of standard version"
	@echo "  debug-hybrid - Debug build of hybrid version"
	@echo "  release      - Optimized build of standard version"
	@echo "  release-hybrid - Optimized build of hybrid version"

.PHONY: all clean test test-hybrid debug debug-hybrid release release-hybrid install uninstall format check setup help hybrid gls
