this is regarding the content for the poster
ok, so a couple notes,
the pipeline flow chart is a realy good idea,
im not a big fan of the "how the algorithm works part tho"
and N3, so lets rework them

the pipeline flow chart is a realy good place to start, we can use it to guide the other figures
+++ well have text files for each figure that will have two secitons, the subtitle  that will be like part of the small text explaination, and then the large text box that will refference them

first we show what were working with, so the next figure will be prety simple, it will be like: original image then divergin arrow to dapi image and to rfp image with the splits like one ontop of the other
( should explain what we mean by standardize intensity)

then we have a figure for the morphological step:
heres a nice wording to describe it:
 We combined information across multiple fluorescence channels and applied morphology based filtering to emphasize structures consistent with the expected crypt appearance. This produced a likelihood map where higher values indicate locations whose intensity and local spatial pattern best match the target profile, even when diffuse staining is present.
i gues the profile is: we know that they are local peaks in the rfp image, theyl be relativly large and have a stable intesity across their area, as well as a more round shape, since their local peaks, their borders will represent the strongest transitions in the image. in high quality images they would have indentations/invaginations but that are much smaller then their total area
and thats all in the rfp channel,
then in the dapi channel which tells us where the cells are and through that where the tissue is, we know that the borders of the crypts are overlapped/ outlined in a U shape by the cells that make them up and excrete the lysozyme  into the open crypt space  the insides of the crypts are theoreticaly cavities that shouldnt show up ( have low to zero dapi signal)

so from the dapi we find the outer border of all the tissue, and the internel cavities

so when we bring them together they define the highest likely hood areas to contain the crypts

we can state this all, kinda in a list

we could use this image to state it realy cleanly, the debug images are realy nice for it /home/ash/documents/code/lysozyme/karens_data/results/higher_quality_images_karen/debug_intermediates/ileum_CH2_7e0c8b

we could show a almost progession from the input images, 
like for tissue dapi, we figure out the outer edges of all the tissue, which includes the interior cavities of the crypts very clearly shown in /home/ash/documents/code/lysozyme/karens_data/results/higher_quality_images_karen/debug_intermediates/ileum_CH2_7e0c8b/identify_crypt_seeds_new/007_tissue_caps_troughs.png saying that this creates a map of the area right outside of the tissue and we look within a range based on how wide we expect the crypts to be

like that could be show original dapi, then => that image with the title of what its elucidating above it

then from the crypt image we pull out the brightest regions within an expected size range to get the best lysozyme positve regions 
then we bring the two results together, which we can show by rendering the tissue_caps_troughs in blue, then overlayin the /home/ash/documents/code/lysozyme/karens_data/results/higher_quality_images_karen/debug_intermediates/ileum_CH2_7e0c8b/identify_crypt_seeds_new/020_good_crypts.png in red basicaly recreating /home/ash/documents/code/lysozyme/karens_data/results/higher_quality_images_karen/debug_intermediates/ileum_CH2_7e0c8b/identify_crypt_seeds_new/021_distance_image.png but in a more human readable manner, and we can say that the areas in the rfp that fall into the cavities or just outside the edges of the dapi image are most likely the lysozyme being secreated( note to verify this may not match reality) into the cavity of the crypt (were punishing rfp signal thats overlapping with tissue) (since were specifying the cavity we can use this to resolve the difference between adjacent crypts wich could have a degree of overlapping signal in the raw rfp image that is chalenging to identify correctly without the appropriate context of where the actualy cavity of the crypt is, since thier can be vallies just due to biological variability that dont represent the borders of crypts) so then once we have identified the continous areas of lysozyme being secreated into the crypt cavities we can use those as seeds unique to each crypt to identify their full representative region. 
( we assume each cavity is unuque to a single crypt)
for that we can show an overlay of the colored seed labels onto a greyscale distance image
( better wording for some part of that, crypt is a concave fold in the epithelium of the intestine, i.e. between vili, it emits lysozyme into the cavity of that fold, from the dapi image we identify a border outside of the epithelium( the epithelium is theoreticaly continous but due ot the 2d sliced nature of imageing we accept that it can be discontinous), and with the rfp image we find where lysozyme is present, if we define crypts even simplier as a continous segment of the epithelium that produce/emits lysozyme: then each crypt will have a distinct location along the border of the epithelium where it emits lysozyme, and so where these two maps overlap to form a continous area there will be the center of a unique crypt) ( as well if we define crypts as a fold in the epithilium of a regular area then we will find that non directly adjacent components of the border will overlap, tho tho not all folds are crypts, we can assume that regions that lack this overlap are at least most likely to be externel)

and this entire figure can probobly be best represented as a flow chart, starting with the two input channels, who arrow to their descriptive versions ( morphologicaly operated), then converge to the colored distance image
then like as a seperate flow we have that distance image ->  greyscale distance image with colored seed labels overlayed -> same image but with the base labels boundries ( and semi transparent bodies(make sure collors match up, seeds 0 transparance, base labels like half)) overlayd
-> just the base labels overlayed onto the og image


then the next diagram can talk about scoring, basicaly we can list out the 5 or so properties, almost like a table, for each show something like /home/ash/documents/code/morphological_animation_toolkit/planned_animation/resources/roi_mt_quality_fixed_hue.png  where we use the hues to represent the scalar quality, and then at the end we have the summed up cumulative qualtiy -> the selections as green or smt outlines
we can add some grafical help, like for cicularity we can literaly draw the circles around them, and for line fit we can draw the line

we should probobly also add a mention of what the actual weight values we used were

basicaly these would be a couple individual diagrams

the current N3 is also kinda useless, we can drop it for now