# AI-Projectwork
Project work of Cognitive Computing and Artificial Intelligence course of University of Catania.

## Specifiche
1. Train the CycleGAN model https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) on the dataset of white and black people's face https://cloud.perceive.dieei.unict.it/index.php/s/6DAy4nKfwCs7ofi). 
CycleGAN generates two models: the first transforms white faces in black faces, the second one transforms white faces in black faces
2. Download the UCF-101 dataset (https://www.crcv.ucf.edu/data/UCF101.php), a dataset of action recognition (input -> video, output -> class).
3. Separate UCF-101 in Utrain, Uval, Utest.
4. Allenate un classificatore sui volti, che data un'immagine del dataset black_white ritorni la classe (bianco vs nero).
5. Utilizzate un modello di face detection (scegliete tra https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/) e generate una nuova versione di Utrain in cui, in ogni immagine, estraete ogni volto, classificate se bianco o nero, e lo convertite nel colore opposto tramite il modello corrispondente in CycleGAN. Generate in questo modo il dataset Utrain-inv.
6. Allenate il classificatore video I3D (https://github.com/piergiaj/pytorch-i3d) su Utrain, e verificate le prestazioni su Utest.
7. Allenate il classificatore video I3D su Utrain-inv, e verificate le prestazioni su Utest.
8. Allenate il classificatore video I3D sull'unione tra Utrain e Utrain-inv, e verificate le prestazioni su Utest.
