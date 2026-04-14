Install Prerequisites:
```    
    # macOS
    brew install cmake gcc libomp

    # Ubuntu / Debian
    sudo apt install cmake build-essential libomp-dev
```

Build:
```
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
```

Compile:
```
    make -j$(nproc)        # Linux
    make -j$(sysctl -n hw.logicalcpu)   # macOS
```

Generate Dataset:
```
    cd ..   # back to VecScale/
    ./build/prepare_dataset \
        --output data/dataset.vsd \
        --num-vectors 100000 \
        --num-queries 1000 \
        --dim 128 \
        --seed 42
```

Run the benchmark:
```
    ./build/run_demo --config configs/default.json
```