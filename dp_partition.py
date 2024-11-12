import time
from typing import List, Tuple

def minimize_difference(setA: List[Tuple[int, int]], setB: List[Tuple[int, int]]) -> Tuple[List[int], List[int], int]:
    # Convert to pairs
    N = len(setA)
    pairs = list(zip(setA, setB))  # Each pair is ((idA, valA), (idB, valB))

    # Convert the float values to integers
    differences = []
    total_diff = 0
    for (idA, valA), (idB, valB) in pairs:
        valA_int = int(valA)
        valB_int = int(valB)
        diff = valA_int - valB_int
        differences.append(diff)
        total_diff += abs(diff)

    # Initialize the DP table
    dp = [{} for _ in range(N + 1)]  # dp[i][s] = (prev_s, choice)
    dp[0][0] = None  # Starting point

    # Build the DP table
    for i in range(1, N + 1):
        di = differences[i - 1]
        dp_i = dp[i]
        dp_prev = dp[i - 1]
        for s in dp_prev:
            # Option 1: Assign di with +1 (valA to set A, valB to set B)
            s_new = s + di
            if s_new not in dp_i:
                dp_i[s_new] = (s, "+")
            # Option 2: Assign di with -1 (valA to set B, valB to set A)
            s_new_neg = s - di
            if s_new_neg not in dp_i:
                dp_i[s_new_neg] = (s, "-")

    # Find the minimal absolute sum
    min_abs_sum = None
    target_s = None
    for s in dp[N]:
        abs_s = abs(s)
        if min_abs_sum is None or abs_s < min_abs_sum:
            min_abs_sum = abs_s
            target_s = s

    # Reconstruct the solution
    ids_setA = []
    ids_setB = []
    s = target_s
    for i in range(N, 0, -1):
        prev_s, sign = dp[i][s]
        ((idA, valA), (idB, valB)) = pairs[i - 1]
        if sign == "+":
            # valA to set A, valB to set B
            ids_setA.append(idA)
            ids_setB.append(idB)
        else:
            # valA to set B, valB to set A
            ids_setA.append(idB)
            ids_setB.append(idA)
        s = prev_s  # Move to the previous state

    # Reverse the IDs to correct the order
    ids_setA.reverse()
    ids_setB.reverse()

    # Compute the total sums to get the final difference
    totalA = 0
    totalB = 0
    setA_ids = set(ids_setA)
    for (idA, valA), (idB, valB) in pairs:
        valA_int = int(valA)
        valB_int = int(valB)
        if idA in setA_ids:
            totalA += valA_int
            totalB += valB_int
        else:
            totalA += valB_int
            totalB += valA_int

    difference = abs(totalA - totalB)
    return ids_setA, ids_setB, difference

# pairs = [(211, 233), (229, 204), (212, 211), (214, 252), (194, 255), (164, 453), (139, 814), (168, 811), (114, 692), (107, 613), (64, 453), (120, 377), (185, 218), (237, 243), (214, 259), (196, 251), (234, 274), (173, 694), (130, 866), (103, 740), (92, 632), (127, 508), (83, 406), (63, 366), (190, 271), (237, 289), (204, 260), (185, 239), (205, 426), (175, 777), (160, 828), (152, 664), (140, 525), (78, 453), (93, 411), (70, 340), (199, 250), (224, 239), (230, 200), (222, 245), (239, 684), (195, 819), (154, 780), (97, 603), (82, 493), (141, 406), (103, 333), (84, 226), (210, 214), (249, 194), (248, 196), (272, 330), (256, 779), (203, 834), (131, 739), (115, 522), (72, 476), (90, 402), (94, 313), (128, 200), (200, 238), (175, 280), (187, 255), (219, 668), (195, 838), (185, 787), (153, 679), (119, 525), (104, 492), (83, 406), (95, 305), (124, 254), (246, 287), (197, 309), (205, 384), (157, 832), (188, 824), (193, 740), (150, 619), (88, 468), (130, 383), (104, 352), (114, 289), (127, 256), (145, 280), (224, 254), (242, 576), (159, 854), (148, 791), (126, 699), (83, 561), (94, 405), (106, 364), (144, 320), (151, 290), (164, 303), (229, 314), (271, 289), (242, 797), (227, 831), (137, 752), (113, 640), (105, 516), (126, 410), (158, 405), (206, 393), (226, 352), (153, 345), (277, 276), (285, 427), (267, 908), (197, 810), (157, 730), (156, 581), (199, 497), (159, 397), (169, 361), (250, 404), (264, 389), (217, 392), (308, 259), (273, 619), (232, 887), (195, 780), (204, 709), (187, 588), (181, 452), (182, 376), (166, 331), (249, 409), (271, 390), (241, 417), (298, 292), (303, 694), (285, 805), (209, 683), (238, 634), (226, 507), (204, 451), (221, 385), (193, 301), (209, 311), (222, 334), (246, 354)]

rawA = [
    (3168, 211),
    (3169, 229),
    (3170, 212),
    (3171, 214),
    (3172, 194),
    (3173, 164),
    (3174, 139),
    (3175, 168),
    (3176, 114),
    (3177, 107),
    (3178, 64),
    (3179, 120),
    (3192, 185),
    (3193, 237),
    (3194, 214),
    (3195, 196),
    (3196, 234),
    (3197, 173),
    (3198, 130),
    (3199, 103),
    (3200, 92),
    (3201, 127),
    (3202, 83),
    (3203, 63),
    (3216, 190),
    (3217, 237),
    (3218, 204),
    (3219, 185),
    (3220, 205),
    (3221, 175),
    (3222, 160),
    (3223, 152),
    (3224, 140),
    (3225, 78),
    (3226, 93),
    (3227, 70),
    (3240, 199),
    (3241, 224),
    (3242, 230),
    (3243, 222),
    (3244, 239),
    (3245, 195),
    (3246, 154),
    (3247, 97),
    (3248, 82),
    (3249, 141),
    (3250, 103),
    (3251, 84),
    (3264, 210),
    (3265, 249),
    (3266, 248),
    (3267, 272),
    (3268, 256),
    (3269, 203),
    (3270, 131),
    (3271, 115),
    (3272, 72),
    (3273, 90),
    (3274, 94),
    (3275, 128),
    (3288, 200),
    (3289, 175),
    (3290, 187),
    (3291, 219),
    (3292, 195),
    (3293, 185),
    (3294, 153),
    (3295, 119),
    (3296, 104),
    (3297, 83),
    (3298, 95),
    (3299, 124),
    (3312, 246),
    (3313, 197),
    (3314, 205),
    (3315, 157),
    (3316, 188),
    (3317, 193),
    (3318, 150),
    (3319, 88),
    (3320, 130),
    (3321, 104),
    (3322, 114),
    (3323, 127),
    (3336, 145),
    (3337, 224),
    (3338, 242),
    (3339, 159),
    (3340, 148),
    (3341, 126),
    (3342, 83),
    (3343, 94),
    (3344, 106),
    (3345, 144),
    (3346, 151),
    (3347, 164),
    (3360, 229),
    (3361, 271),
    (3362, 242),
    (3363, 227),
    (3364, 137),
    (3365, 113),
    (3366, 105),
    (3367, 126),
    (3368, 158),
    (3369, 206),
    (3370, 226),
    (3371, 153),
    (3384, 277),
    (3385, 285),
    (3386, 267),
    (3387, 197),
    (3388, 157),
    (3389, 156),
    (3390, 199),
    (3391, 159),
    (3392, 169),
    (3393, 250),
    (3394, 264),
    (3395, 217),
    (3408, 308),
    (3409, 273),
    (3410, 232),
    (3411, 195),
    (3412, 204),
    (3413, 187),
    (3414, 181),
    (3415, 182),
    (3416, 166),
    (3417, 249),
    (3418, 271),
    (3419, 241),
    (3432, 298),
    (3433, 303),
    (3434, 285),
    (3435, 209),
    (3436, 238),
    (3437, 226),
    (3438, 204),
    (3439, 221),
    (3440, 193),
    (3441, 209),
    (3442, 222),
    (3443, 246),
]

rawB = [
    (588, 233),
    (589, 204),
    (590, 211),
    (591, 252),
    (592, 255),
    (593, 453),
    (594, 814),
    (595, 811),
    (596, 692),
    (597, 613),
    (598, 453),
    (599, 377),
    (612, 218),
    (613, 243),
    (614, 259),
    (615, 251),
    (616, 274),
    (617, 694),
    (618, 866),
    (619, 740),
    (620, 632),
    (621, 508),
    (622, 406),
    (623, 366),
    (636, 271),
    (637, 289),
    (638, 260),
    (639, 239),
    (640, 426),
    (641, 777),
    (642, 828),
    (643, 664),
    (644, 525),
    (645, 453),
    (646, 411),
    (647, 340),
    (660, 250),
    (661, 239),
    (662, 200),
    (663, 245),
    (664, 684),
    (665, 819),
    (666, 780),
    (667, 603),
    (668, 493),
    (669, 406),
    (670, 333),
    (671, 226),
    (684, 214),
    (685, 194),
    (686, 196),
    (687, 330),
    (688, 779),
    (689, 834),
    (690, 739),
    (691, 522),
    (692, 476),
    (693, 402),
    (694, 313),
    (695, 200),
    (708, 238),
    (709, 280),
    (710, 255),
    (711, 668),
    (712, 838),
    (713, 787),
    (714, 679),
    (715, 525),
    (716, 492),
    (717, 406),
    (718, 305),
    (719, 254),
    (732, 287),
    (733, 309),
    (734, 384),
    (735, 832),
    (736, 824),
    (737, 740),
    (738, 619),
    (739, 468),
    (740, 383),
    (741, 352),
    (742, 289),
    (743, 256),
    (756, 280),
    (757, 254),
    (758, 576),
    (759, 854),
    (760, 791),
    (761, 699),
    (762, 561),
    (763, 405),
    (764, 364),
    (765, 320),
    (766, 290),
    (767, 303),
    (780, 314),
    (781, 289),
    (782, 797),
    (783, 831),
    (784, 752),
    (785, 640),
    (786, 516),
    (787, 410),
    (788, 405),
    (789, 393),
    (790, 352),
    (791, 345),
    (804, 276),
    (805, 427),
    (806, 908),
    (807, 810),
    (808, 730),
    (809, 581),
    (810, 497),
    (811, 397),
    (812, 361),
    (813, 404),
    (814, 389),
    (815, 392),
    (828, 259),
    (829, 619),
    (830, 887),
    (831, 780),
    (832, 709),
    (833, 588),
    (834, 452),
    (835, 376),
    (836, 331),
    (837, 409),
    (838, 390),
    (839, 417),
    (852, 292),
    (853, 694),
    (854, 805),
    (855, 683),
    (856, 634),
    (857, 507),
    (858, 451),
    (859, 385),
    (860, 301),
    (861, 311),
    (862, 334),
    (863, 354),
]

# elementsA = [x[1] for x in rawA]
# elementsB = [x[1] for x in rawB]
# print(elementsA)
# print(elementsB)


start = time.time()
A, B, diff = minimize_difference(rawA, rawB)
end = time.time()
print("Time:", end - start)
print("A:", A)
print("B:", B)
print("Difference:", diff)
print("Maximum sum:", max(sum(A), sum(B)))
