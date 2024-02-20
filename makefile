# Compiler
CC = g++
CXX = $(CC)

# OSX include paths 
CFLAGS = -Wc++11-extensions -std=c++11 -I./include -DENABLE_PRECOMPILED_HEADERS=OFF $(shell pkg-config --cflags opencv4)

# Dwarf include paths
CXXFLAGS = $(CFLAGS)

# opencv libraries
LDLIBS = $(shell pkg-config --libs opencv4)

BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./include

# Target exe
TARGET = $(BINDIR)/obj_recog

# Source files
SRCS = $(wildcard $(SRCDIR)/*.cpp)

# Object files
OBJS = $(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

# Ensure the output directory exists
$(shell mkdir -p $(BINDIR) $(OBJDIR))

# Build the target
$(TARGET): $(OBJS)
	$(CC) $^ -o $@.exe $(LDLIBS)

# # Linking executable to object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@ 

# Include dependencies
-include $(OBJS:.o=.d)

# Generate dependencies
$(OBJDIR)/%.d: $(SRCDIR)/%.cpp
	$(CC) $(CXXFLAGS) -MM -MT $(@:.d=.o) $< > $@

# Clean up
clean:
	rm -f $(OBJDIR)/*.o $(TARGET)

# Phony targets - will run regardless of file existence
.PHONY: clean