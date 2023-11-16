class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./mnist", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False)
        self.mnist_predict = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return TripletDataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return TripletDataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return TripletDataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return TripletDataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # NOTE(liamvdv): used to clean-up when the run is finished
        pass
