# Import exist and generator
from os import remove
from os.path import exists, join
from pathlib import Path
from Datasets.Generator import Generator

from ML.TensorFlow import TensorFlow # Import tensorflow ml class
                # @todo: Remove and import/create dynamically depending on dict (as we can try different ML libaries)

cleanDatasets = False
pathPre = Path("Datasets")
pathTraining = pathPre / Path("Training.csv")
pathValidation = pathPre / Path("Validation.csv")
pathTesting = pathPre / Path("Testing.csv")

def main():
    # Delete previous datasets
    if cleanDatasets:
        if exists(pathTraining):
            remove(pathTraining)
        if exists(pathValidation):
            remove(pathValidation)
        if exists(pathTesting):
            remove(pathTesting)

    # Check if the datasets has been formatted in training, validation, testing
    if not (exists(pathTraining) and exists(pathValidation) and exists(pathTesting)):
        generator = Generator(["Datasets/True.csv", "Datasets/Fake.csv"])
        generator.mTables["True"]["label"] = 1
        generator.mTables["Fake"]["label"] = 0
        generator.combineTables(shuffle=True)
        generator.exportCTable(path=pathPre, split=(0.8, 0.9, 1))
    pathtraining = Path("C:\Projects\FakeNewsML\Datasets\Training.csv")
    ml = TensorFlow(pathtraining, pathValidation, pathTesting)
    # @todo: Below functions
    ml.loadModel()
    ml.fitModel()
    # ml.saveModel()
    # ml.evaluate()
    # ml.predict()


    return


if __name__ == "__main__":
    main()