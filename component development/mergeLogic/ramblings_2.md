# merge logic
- the critical part is that we arent always going to merge, alot of the time the best option is the null/ mapping to self
    - this means that we are going to need comprable parameters to compare to the grouping options:
        - first: "niegbor bound perimiter" this will honeslty need to be a manual paremeter that primaryly serves to exclude those tiny regions of contact between irl seperate crypts
            - can be manualy set, but a good starting value would be /5
            - tho their of course are scenarios where a tiny protrusion has low perimeter overlap but does in fact belong to a neigboring labels crypts, we could account for this by factorind in the delta surface area with the perimter 
                - could do group area / label area => bigger group area = larger then 1
                - x shared perimeter would favor greater shared area then /total perimter of the label would convert that to a preprotion of the labels area
                - this way singletons would essential be 1 x standard factor vrs groupigns which would be a ratio representing the increase in area x the total shared perimeter
                - 

- steps
    - first which parings of direct neighbors are best
        - net surface perimiter shared by neighbors
            - positive factors
                - more shared perimeter => more complete wrapping by neigbors
            - negative factors
                - surface area of subject label not covered by neigbors => prioritize more complete groupings
    - second which pairs of secondary neighbors are best
        - larger grouping => often better
            - surface area = positive variable
        - consolidated => deselect stringy and spread out combinations
            - surface area/perimeter = sellects for more consolidated groupings
        - average distance of label from center of mass of grouping
            - avoids /0 from delta center of masss and get scaling difference for dif sized label areas
            - that way always get a >0 value that increases as center moves away from the label
        - total shared perimiter vrs unshared perimeter
            - more would be better


- need to be able to collapse down label assignnts
    - teirs
        - inital labels -> neigbor 2 neigbor mappings -> potential group mappings -> group to primary label mappings
    - the group to primary label mappings needs to be collapsed i.e.:
        - init group A -> label A, group B -> label B
            - but then label B gets assinged to group A, so then group B should be collapsed to label A






can you detect starvation by analyzing urine for protien metabolisis byproducts, get like a reactive liter

go to apps.yenlab.nutrisci.wisc.edu/colostrum_rna/




order of analysis for optimization

- first get neigbor list, then for each label get all groupings of directly connected labels
    - idealy we would be able to throw out a bunch of potential combinations before we have to  evaluate the second parameter
    - assume we have a dict of all mappings
    - for each label that we are evaluating, grab sets of other lables nighboring it grouped by which ones are in contact with one another
