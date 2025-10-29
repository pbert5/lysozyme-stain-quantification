basicaly with my lysozyme segmentation algo, i have this consistant issue where it will either oversegment by one or sometimes merge a bunch of adjacent crypts, with the new algos i have minimized it but it still hapens, which is a problem bc i expect an exact number of crypts for the math, so i was thinking we could stop gap it by creating a "how many crypts" metric thats calculated on the final label array of crypts to try and gues how many crypts were actualy segmented, since for the simple lysozyme out as long as the numberof crypts is correct even if we have merging it wont mess up the math

1. first we assume all crypts are in a 1d line, and not in a 2d array, such that no given crypt is entierly surrounded by other crypts: it could share up to two side but two apposing sides will be free
we could aproach it through a group connensous, if we assume that for a given detection the positive hypothesis is that a single acurately segmented crypt that is representative of the morphology and scale of the population of true crypts within the image we want to detect, and that all other correct detections should have a similar morphology and scale to our given crypt, we could judge all other detections on the grounds of how similar they are to the given crypt
for scenarios where multiple crypts are detected as a single crypt and assumption1 is correct: the morphological properties of the component crypts are still derived from the "true crypts" within the image, so they will individualy have a similar area and shape, but will apeare merged on one of two apposing sides, thus their area and shape will be related to the packing density of a given component crypt
2. by observation merged crypts tend to be a singular occurance

so assuming a given detection(alpha) is of a single crypt, all other detections will either be matching detecitons of single crypts, detecitons of multiple merged crypts whos likely quantity can be defined by the non overlapping packing density of our given crypt within that labeld region, or partial detections that subdivided a given crypt, who will likely represent fractions of alpha

for the fractions we have two options:
    1. either try to figure out what fraction of alpha it is to report for the count which coould allow over merges to have a negating effect on our count
    2. or see if there is a more complete pairing of detecitons that matches alpha better limiting to adjacencies
wheter it can find a appropriate merger then it would prepose preforming that, otherwise: nill

we can probobly prevent over segmented crypts ( assumeing regular high severity is 1/2 to 1/3, and that all true crypts share the same orientation(or have been transformed to do so))  by just not rotating the detections to fit, i.e. for a given 1/2 of a circle you will only be able to fit one instance into the full circle without allowing for rotation

essentialy if each detection assumes that it is perfect, and compares its self to the other detections, they can vote on 
    a. how many crypts they "believe" the other deteciton contain
    b. effective mergers to resolve oversegmentation
a given detection will beleive it contains a single crypt

assume a good algorithm has detections primarily composed of singluar crypts, while the few merged detections will contain between 1 and many

and if we assume their are usualy more then 5 crypts per image and we can cherry pick the best 5 crypts, we can skip noise

we can score all of the "best" detections based on quality metrics to weigh their contribuitions to the final crypt count
and then by basiclay just averagin out we have an estimated real # number of crypts detected as well as number of crypts per detection