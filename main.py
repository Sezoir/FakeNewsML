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

    # pathtraining = Path("C:\Projects\FakeNewsML\Datasets\Training.csv")
    ml = TensorFlow(pathTraining, pathValidation, pathTesting)
    # @todo: Below functions
    ml.loadModel()
    # ml.fitModel()
    # ml.saveModel()
    # ml.evaluate()
    # Should be true (taken from testing dataset)
    ml.predict("Pope denounces healthcare inequality in rich countries", "VATICAN CITY (Reuters) - Pope Francis condemned on Thursday inequality in healthcare, particularly in rich countries, saying governments had a duty to ensure the common good for all its citizens.  Increasingly sophisticated and costly treatments are available to ever more limited and privileged segments of the population,  Francis said in an address to a conference of European members of the World Medical Association.  This raises questions about the sustainability of healthcare delivery and about what might be called a systemic tendency toward growing inequality in healthcare,  he said. The tendency was clearly apparent when you compared healthcare cover between countries and continents, the pope said, adding that it was also visible within more wealthy countries,  where access to healthcare risks being more dependent on individuals  economic resources than on their actual need for treatment.  Francis did not mention any countries. Healthcare is a big issue in the United States, where President Donald Trump has vowed to get rid of the Affordable Care Act, introduced by his predecessor, Barack Obama, which aimed to make it easier for lower-income households to get health insurance. He said healthcare legislation needed a  broad vision and a comprehensive view of what most effectively promotes the common good in each concrete situation.  In speaking of end-of-life issues, Francis re-affirmed the Catholic Church s long-standing teaching that it is morally acceptable for a patient or a family to suspend or reject  disproportionate measures  to keep a terminally ill person alive. But he stressed that this was  different from euthanasia, which is always wrong, in that the intent of euthanasia is to end life and cause death . Regarding end-of-life decisions, the pope said governments had a duty  to protect all those involved, defending the fundamental equality whereby everyone is recognized under law as a human being living with others in society .")
    # Should be false (taken from dataset)
    ml.predict("KID ROCK GOES OFF On Colin Kaepernick, Obamacare and Deadbeat Dads…Asks “Almighty Jesus” To “Give us strength to fight”…”Because like it or not, Hillary LOST and your President Is Donald Mother-F***king Trump!” During MI Concert",
               "Kid Rock hasn t officially announced his run for US Senate, but every day he comes increasingly closer. During a concert in Grand Rapids, MI, Kid Rock took time to get very political, as he addressed some serious issues that he is clearly passionate about.Kid Rock started out by addressing his Grand Rapids, MI audience:  What s going on in the world today? It seems the government wants to give everyone health insurance but wants us all to pay. And to be very frank, I don t really don t have a problem with that, because God has blessed me, and made my pockets fat.  But redistribution of wealth seems more like their plan. And I don t believe that you should say sacrifice, do things by the book and then take care of some deadbeat, milking the system, lazy ass motherf -. Rock went on to address single moms who keep having kids to get more welfare. His most passionate statement was against deadbeat dads who refuse to be a  man  and raise their children. Rock has a little experience in that area, as he raised his black son as a single father. His son attended a private rural Catholic school and graduated from college. Robert James Ritchie, Jr. is married, and in 2015, he and his wife gave Rock his first grandchild, Skye Noelle.Lastly, Robert Ritchie aka Kid Rock took on the left who have been attacking him for criticizing Colin Kaepernick for taking a knee during the national anthem. He put the left on notice that he won t be sitting back and allowing them to label him a  racist  or a  Nazi. Here s the uncut (shortened) version: ***Language warning***Watch as Kid Rock is introduced as  The next great Senator of the great state of Michigan. ")
    # Should be true (found of foxnews)
    ml.predict("Ghislaine Maxwell indictment: How Jeffrey Epstein's alleged madam groomed his victims",
               "Jeffrey Epstein's longtime confidant and former girlfriend Ghislaine Maxwell was arrested Thursday and is facing multiple sex abuse charges stemming from an alleged sex trafficking conspiracy involving underage girls. In a six-count indictment unsealed after her arrest, prosecutors allege Maxwell enticed minors to travel to Epstein's multiple residences-- including a multi-story apartment on Manhattan's Upper East Side, his residence in Palm Beach, Fla., his ranch in Santa Fe, New Mexico, and to Maxwell's house in London-- to engage in illegal sex acts")


    return


if __name__ == "__main__":
    main()