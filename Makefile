NVCC = nvcc
TARGET = hello
SRC = hello.cu

all: $(TARGET) ptx sass

$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET)

ptx: $(SRC)
	$(NVCC) -ptx $(SRC) -o hello.ptx

sass: $(SRC)
	$(NVCC) -cubin $(SRC) -o hello.cubin
	nvdisasm hello.cubin > hello.sass

clean:
	rm -f $(TARGET) hello.ptx hello.sass hello.cubin
