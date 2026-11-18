# CRDIC Streamlit Community Cloud demo

This repository hosts the Streamlit entry point for the CRDIC distributed-cloud POC.

The app downloads the public clean checkpoint, 50 target images, 50 side-information images, and the FAISS index from the Hugging Face Space at:
https://huggingface.co/spaces/heiheilxr/distributed_cloud_coding_test

In Streamlit Community Cloud, select `streamlit_cloud_app.py` as the main file. The first launch downloads and caches the bundle, then exposes target-only reconstruction, FAISS side-information retrieval, the threshold decision, tau control, PSNR, LPIPS, estimated BPP, and timing.

The free CPU deployment does not compile the native VTM encoder; the VVC panel is therefore explicitly marked unavailable.
