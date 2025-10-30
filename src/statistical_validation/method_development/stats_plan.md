# core rules
- we will be using r for the analysis
- using the dplyr and purr and tibil packages for analaysis
- prioritize using "|>" pipes where ever posible as apposed to anythign nested
# preperation
## manual data preperation
- formating
    - first off we need to format the manual dat, theres a weired thing where theres a bunch of dups stacked right next to each other, so well basicaly just collapse duplicate coll names, group by slize and average out, + remove empty data
## automated data preperation
- parsing subject names
    - first 4-5 characters before \- are the mouse id
    - will be grouping by pathing in the \[\], obv will be grouping by string before last \/
        - for comparisons, will assum that the manual analyssi were all fo the same groups so will map by groups
        a. basicaly the critial parts are like: is it from Jej LYZ, but we will be collapsing together the /G3 ir /G2 parts
        b. for retakes, alot of the time its formated like this: Retake G2 - ABX in which case we would just be grouping by the retake part, so
        c. the regex would look ike, first is their a "/" then take the part preceding that, if not is there a space then take up tot he first space unless its lysozyme retakes2.0, then jsut take that as a group
    - this willl tell us the image sets
- we will also start by only using the auto results that were ranked as pass in /home/phillip/documents/lysozyme/results/simple_dask/manual_verification_ratings.csv

## cross source mapping
per manual quant csv: 
    1. append duplicate coll names verticaly, propate approprate mouse id to all new rows
    2. group by slice
    3. try both
    a.
        3. collapse Average Size  and Mean(flouressence) by averageing, note std

        4. join seperate person manaula analsysis %%name in \(\) at end of file name%% make sure slices match up and join by either averagi
    b. 
        3. join sep manaul analysis verticaly
        4. collapse Average Size  and Mean(flouressence) by averageing, note std, also get attribuites of count and total area
per automated quant
    1. the meta data in brackets is important, the part before the last \/ will tell us which set of images they come from, theoreticaly the manual analsis is in terms of originals + maybe retakes



now that we will have both of these set up, out first tier of mapping will be:
    each auto proccessed image set against the combined manual analysis


now we map automated analysis to manual analysis:
we have two options and were going to test both:
a. exact image maps, where  we take the slice strings from manaual analsis and find the images in auto analysis that match, ( obv preproc, will need ot remove the .tif part)  will prioritize coming from origianls, then retakes, then lysoszyme retakes 2.0
b. by animal id : from auto analsys when we have seperate iamges of the same subject ( i.e. they have the same animal id and come from the same image set we collapse them togetether first before we compare)




## result mapping
- total area and average size should theoreticaly be on the same scale across the manual to automated, but i suspect that it wont be so were going to need to solve for that:
    - first option is going after the % area metric in the manaul analysis, get the total area of the coresponding images in both mpp and pixels and see which is which (image tot area )*\% area is closer to the reported total areas and that may tell us what they are using %% can get scale from /home/phillip/documents/lysozyme/results/simple_dask/simple_dask_image_summary_detailed.csv

# stats
- were going to  want to compare my automated results to the manaual results, 
 At the highest level,  claim is:

“The automated quantification results agree with (or accurately reproduce) the manual measurements performed by human annotators.”

That means:

Automated Total Area ≈ Manual Total Area

Automated Average Size ≈ Manual Average Size

Automated Mean Fluorescence ≈ Manual Mean Fluorescence


