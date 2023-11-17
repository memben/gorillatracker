# gorillatracker
The GorillaTracker BP


# Architecture
### Adding a Dataset
1. create a Dataset that supports __getitem__(self, idx: int) (read: single element access) in `gorillatracker.dataset_<name>.py`. 
2. create a subclass from `gorillatracker.data_modules.TripletDataModule` and `gorillatracker.data_modules.QuadletDataModule` in `gorillatracker.data_module_<name>.py`, implement `get_dataset_class` and `get_transforms`.

You can now use the data module exposed in `gorillatracker.data_module_<name>.py` for online and offline triplet loss. All the sampling for triplet generation is ready build. 


# Statistcs

```bash
ground_truth/ 
  cxl/ contains 952 images of 120 individuals  
```
