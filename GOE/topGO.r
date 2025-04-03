# Rscript1.R
# eg. run: 
# Rscript Rscript1.R annotations.txt interestinggenes.txt output_file

args <- commandArgs(trailingOnly = TRUE)
GOfile = args[1]
refsetFile = args[2]
interestingGenesFile = args[3]
Pvalue = args[4]
GOtype = args[5]   #BP, MF and CC
output_file = args[6]

# set the output file
sink(output_file)

# load topGO
library("topGO")

# read in the 'gene universe' file
geneID2GO <- readMappings(file = GOfile)

#read in reference set
refset <- read.table(refsetFile,header=FALSE)
geneUniverse <- as.character(refset$V1)


# read in the genes of interest 
genesOfInterest <- read.table(interestingGenesFile,header=FALSE)
genesOfInterest <- as.character(genesOfInterest$V1)


## make a geneList factor object using refset and testset
geneList <- factor(as.integer(geneUniverse %in% genesOfInterest))
names(geneList) <- geneUniverse

# build the GOdata object in topGO
myGOdata <- new("topGOdata", description="My project", ontology=GOtype, allGenes=geneList,  annot = annFUN.gene2GO, gene2GO = geneID2GO)
myGOdata

# run the Fisher's exact tests
resultClassic <- runTest(myGOdata, algorithm="classic", statistic="fisher")
resultElim <- runTest(myGOdata, algorithm="elim", statistic="fisher")
resultTopgo <- runTest(myGOdata, algorithm="weight01", statistic="fisher")
resultParentchild <- runTest(myGOdata, algorithm="parentchild", statistic="fisher")
 
# see how many results we get where weight01 gives a P-value <= Pvalue:
mysummary <- summary(attributes(resultTopgo)$score <= Pvalue)
numsignif <- as.integer(mysummary[[3]]) # how many terms is it true that P <= Pvalue



if (numsignif==0)
{
	cat("#Significant GO terms: ", numsignif)
	quit()
}


# print out the top 'numsignif' results:
allRes <- GenTable(myGOdata, topgoFisher = resultTopgo, classicFisher = resultClassic, elimFisher = resultElim, parentchildFisher = resultParentchild, orderBy = "topgoFisher", ranksOf = "topgoFisher", topNodes = numsignif)
allRes


# print a graph (to a pdf file) with the top 'numsignif' results:
output_file2 = paste(output_file,"Topgo", sep="_")
printGraph(myGOdata, resultTopgo, firstSigNodes = numsignif, fn.prefix = output_file2, useInfo = "all", pdfSW = TRUE)
 
# print out the genes that are annotated with the significantly enriched GO terms:
myterms <- allRes$GO.ID
mygenes <- genesInTerm(myGOdata, myterms)
for (i in 1:length(myterms))
{
   myterm <- myterms[i]
   mygenesforterm <- mygenes[myterm][[1]] 
   myfactor <- mygenesforterm %in% genesOfInterest # find the genes that are in the list of genes of interest
   mygenesforterm2 <- mygenesforterm[myfactor == TRUE] 
   mygenesforterm2 <- paste(mygenesforterm2, collapse=',')
   print(paste("Term",myterm,"genes:",mygenesforterm2))
}
# close the output file
sink()
