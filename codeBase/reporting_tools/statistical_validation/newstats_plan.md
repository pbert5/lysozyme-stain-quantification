thats some great initial investigation, tho now we need to clean it up into a script thats easily auditable and returns the summarized data to be externely analyzid with pearson or whatever, so what i did was moove all of the old stuff into /home/phillip/documents/lysozyme/src/statistical_validation/method_development since it was kindof a mess and we are going to make one clean easily auditable final script whos primary output will be one consolidated csv for easy externel analsysis,


its collumns will be: mouse id, manual slice source, manual slice name, measured average flourecences, number of manual crypts, autosubject name ( we will not be collapsing between subject yet, so basicaly the previous stuff can just be duplicate for each unique manual to auto mapping ),  + of course auto source type
 
tho for auditing we will output somethings with alot more detail along the way:
for manual analysis:
we will have a cleaned up but uncolapsed aggregation of the manual data, basicaly just a couple pivits within each manual csv to split up and stack together the duplicate manual collums ( oc manualy make sure that  adjacency parrings between average size and mean dont get mixed up, (also turns out these replicates are diff crypts within the image and the "count" metric is just an interal oversegmentation that they were able to ignore by brute force) and ocourse append the seperate ones together just through sources into thier own col, we can also at this point add a flourecense per crypt metric col by just multiplying total area by mean tho total area and mean are in terms of pixels,  and i see some 40x'ers so we will need to convert them following the same name to mpp logic from /home/phillip/documents/lysozyme/src/lysozyme_pipelines/pipeline.py ( the keys and scales will need to be a var explicitly set in the beggining of the script)

then we will have our manual collapsed dataset that we will get by grouping by manual source csv and slice ( so that we operate within individual manual analysis), where we will get the average and sd of our per crypt flourcence value for a avgImgFlourcence value, essentialy for each analysis of each crypt, will also need to note meta stuff like num crypts = count of rows in end group

then we figure out the mappings from manual to automatic: we kindof already had a good system for that but jus tmake sure that we are comparing manual slice name to autosubject name, for the mappings can basicaly ignore manual source and just figure out how the names map together
the once you have the mappings built together, basicaly just append them back with the manual collapsed data set along with their coresponding auto flourecence results/ crypts count or whatever other outo results




