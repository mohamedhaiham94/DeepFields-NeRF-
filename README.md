# NeRF
3DVolumeReconstruction

---
### Setting Up the Python Virtual Environment (Python 3.11)

**1. Create a Virtual Environment**
```python
python -m venv .venv
```
**2. Activate the Virtual Environment**
```python 
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate
```

**3. Install Dependencies**
```python
pip install -r requirements.txt
```
To use pytorch, I am using torch==2.6.0 with CUDA 12.6.

We also have to install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) framework to use the [instant-ngp](https://github.com/NVlabs/instant-ngp).

```python
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

---