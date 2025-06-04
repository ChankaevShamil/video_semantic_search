# video_semantic_search

This app is capable of conducting a semantic video search based on a text query.

This app load video by YouTube link in mp4 format.
MP4 - is format for containing compress video. Instead contain all frames, this codec contain first frame and diff for the next frames.
This first containing frame called intra frame (I-frame).
So, we extract all I-frames, using PyAV. 
After this, we use CLIP to get embeddings for this I-frames, and create vector database, using FAISS.
User text query also transform in emeddings, using CLIP. After this, app extracting k most similar frames and return timestamps for this in moments.


We recommended launch this app by docker: docker compose up --build
