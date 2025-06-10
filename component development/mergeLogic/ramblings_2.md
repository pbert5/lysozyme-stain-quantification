# merge logic
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
