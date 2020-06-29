from os.path import exists
import Datasets.dataGenerator as gen

def main():
    # Check if the datasets has been formatted in training, validation, testing
    if not (exists("Datasets/Training.csv") and exists("Datasets/Validation.csv") and exists("Datasets/Testing.csv")):
        generator = gen.Generator(["Datasets/True.csv", "Datasets/Fake.csv"])
        generator.mTables["True"]["label"] = 1
        generator.mTables["Fake"]["label"] = 0
        generator.combineTables(shuffle=True)
        generator.exportCTable(path="Datasets/", split=(0.8, 0.9, 1))


    return


if __name__ == "__main__":
    main()