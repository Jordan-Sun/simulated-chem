from typing import List, Tuple
import itertools
import numpy as np

def minimize_difference(elements: List[Tuple[int, float]]) -> Tuple[List[int], List[int], float]:
    # Initialize DP table: dp[s][k] = True if sum s is achievable with k elements
    N = len(elements) // 2
    total_sum = sum(value for _, value in elements)
    target = total_sum // 2
    # dp[k] is a set of sums achievable with k elements
    dp = [set() for _ in range(N+1)]
    # backtrace[k][s] = element that led to sum
    backtrace = [{} for _ in range(N+1)]
    # Base case: sum 0 achievable with 0 elements
    dp[0].add(0)

    for element in elements:
        for k in range(N, 0, -1):
            for s in dp[k-1]:
                new_sum = s + element[1]
                if new_sum > target:
                    break   # No need to consider larger sums
                if new_sum in dp[k]:
                    continue # Already found a subset summing to new_sum
                dp[k].add(new_sum)
                backtrace[k][new_sum] = element

    # Find the largest sum s achievable with N elements
    best_sum = max(dp[N])
    # Compute the minimum difference
    min_diff = total_sum - 2*best_sum
    # Reconstruct the subset summing to best_sum
    set_A = []
    for k in range(N, 0, -1):
        element = backtrace[k][best_sum]
        set_A.append(element)
        best_sum -= element[1]

    # The other set is the elements not in set_A
    set_B = [a for a in elements if a not in set_A]

    return set_A, set_B, min_diff

# Time the function
import time

elements = [(3168, np.float64(211.0)), (3169, np.float64(229.0)), (3170, np.float64(212.0)), (3171, np.float64(214.0)), (3172, np.float64(194.0)), (3173, np.float64(164.0)), (3174, np.float64(139.0)), (3175, np.float64(168.0)), (3176, np.float64(114.0)), (3177, np.float64(107.0)), (3178, np.float64(64.0)), (3179, np.float64(120.0)), (3192, np.float64(185.0)), (3193, np.float64(237.0)), (3194, np.float64(214.0)), (3195, np.float64(196.0)), (3196, np.float64(234.0)), (3197, np.float64(173.0)), (3198, np.float64(130.0)), (3199, np.float64(103.0)), (3200, np.float64(92.0)), (3201, np.float64(127.0)), (3202, np.float64(83.0)), (3203, np.float64(63.0)), (3216, np.float64(190.0)), (3217, np.float64(237.0)), (3218, np.float64(204.0)), (3219, np.float64(185.0)), (3220, np.float64(205.0)), (3221, np.float64(175.0)), (3222, np.float64(160.0)), (3223, np.float64(152.0)), (3224, np.float64(140.0)), (3225, np.float64(78.0)), (3226, np.float64(93.0)), (3227, np.float64(70.0)), (3240, np.float64(199.0)), (3241, np.float64(224.0)), (3242, np.float64(230.0)), (3243, np.float64(222.0)), (3244, np.float64(239.0)), (3245, np.float64(195.0)), (3246, np.float64(154.0)), (3247, np.float64(97.0)), (3248, np.float64(82.0)), (3249, np.float64(141.0)), (3250, np.float64(103.0)), (3251, np.float64(84.0)), (3264, np.float64(210.0)), (3265, np.float64(249.0)), (3266, np.float64(248.0)), (3267, np.float64(272.0)), (3268, np.float64(256.0)), (3269, np.float64(203.0)), (3270, np.float64(131.0)), (3271, np.float64(115.0)), (3272, np.float64(72.0)), (3273, np.float64(90.0)), (3274, np.float64(94.0)), (3275, np.float64(128.0)), (3288, np.float64(200.0)), (3289, np.float64(175.0)), (3290, np.float64(187.0)), (3291, np.float64(219.0)), (3292, np.float64(195.0)), (3293, np.float64(185.0)), (3294, np.float64(153.0)), (3295, np.float64(119.0)), (3296, np.float64(104.0)), (3297, np.float64(83.0)), (3298, np.float64(95.0)), (3299, np.float64(124.0)), (3312, np.float64(246.0)), (3313, np.float64(197.0)), (3314, np.float64(205.0)), (3315, np.float64(157.0)), (3316, np.float64(188.0)), (3317, np.float64(193.0)), (3318, np.float64(150.0)), (3319, np.float64(88.0)), (3320, np.float64(130.0)), (3321, np.float64(104.0)), (3322, np.float64(114.0)), (3323, np.float64(127.0)), (3336, np.float64(145.0)), (3337, np.float64(224.0)), (3338, np.float64(242.0)), (3339, np.float64(159.0)), (3340, np.float64(148.0)), (3341, np.float64(126.0)), (3342, np.float64(83.0)), (3343, np.float64(94.0)), (3344, np.float64(106.0)), (3345, np.float64(144.0)), (3346, np.float64(151.0)), (3347, np.float64(164.0)), (3360, np.float64(229.0)), (3361, np.float64(271.0)), (3362, np.float64(242.0)), (3363, np.float64(227.0)), (3364, np.float64(137.0)), (3365, np.float64(113.0)), (3366, np.float64(105.0)), (3367, np.float64(126.0)), (3368, np.float64(158.0)), (3369, np.float64(206.0)), (3370, np.float64(226.0)), (3371, np.float64(153.0)), (3384, np.float64(277.0)), (3385, np.float64(285.0)), (3386, np.float64(267.0)), (3387, np.float64(197.0)), (3388, np.float64(157.0)), (3389, np.float64(156.0)), (3390, np.float64(199.0)), (3391, np.float64(159.0)), (3392, np.float64(169.0)), (3393, np.float64(250.0)), (3394, np.float64(264.0)), (3395, np.float64(217.0)), (3408, np.float64(308.0)), (3409, np.float64(273.0)), (3410, np.float64(232.0)), (3411, np.float64(195.0)), (3412, np.float64(204.0)), (3413, np.float64(187.0)), (3414, np.float64(181.0)), (3415, np.float64(182.0)), (3416, np.float64(166.0)), (3417, np.float64(249.0)), (3418, np.float64(271.0)), (3419, np.float64(241.0)), (3432, np.float64(298.0)), (3433, np.float64(303.0)), (3434, np.float64(285.0)), (3435, np.float64(209.0)), (3436, np.float64(238.0)), (3437, np.float64(226.0)), (3438, np.float64(204.0)), (3439, np.float64(221.0)), (3440, np.float64(193.0)), (3441, np.float64(209.0)), (3442, np.float64(222.0)), (3443, np.float64(246.0)), (588, np.float64(233.0)), (589, np.float64(204.0)), (590, np.float64(211.0)), (591, np.float64(252.0)), (592, np.float64(255.0)), (593, np.float64(453.0)), (594, np.float64(814.0)), (595, np.float64(811.0)), (596, np.float64(692.0)), (597, np.float64(613.0)), (598, np.float64(453.0)), (599, np.float64(377.0)), (612, np.float64(218.0)), (613, np.float64(243.0)), (614, np.float64(259.0)), (615, np.float64(251.0)), (616, np.float64(274.0)), (617, np.float64(694.0)), (618, np.float64(866.0)), (619, np.float64(740.0)), (620, np.float64(632.0)), (621, np.float64(508.0)), (622, np.float64(406.0)), (623, np.float64(366.0)), (636, np.float64(271.0)), (637, np.float64(289.0)), (638, np.float64(260.0)), (639, np.float64(239.0)), (640, np.float64(426.0)), (641, np.float64(777.0)), (642, np.float64(828.0)), (643, np.float64(664.0)), (644, np.float64(525.0)), (645, np.float64(453.0)), (646, np.float64(411.0)), (647, np.float64(340.0)), (660, np.float64(250.0)), (661, np.float64(239.0)), (662, np.float64(200.0)), (663, np.float64(245.0)), (664, np.float64(684.0)), (665, np.float64(819.0)), (666, np.float64(780.0)), (667, np.float64(603.0)), (668, np.float64(493.0)), (669, np.float64(406.0)), (670, np.float64(333.0)), (671, np.float64(226.0)), (684, np.float64(214.0)), (685, np.float64(194.0)), (686, np.float64(196.0)), (687, np.float64(330.0)), (688, np.float64(779.0)), (689, np.float64(834.0)), (690, np.float64(739.0)), (691, np.float64(522.0)), (692, np.float64(476.0)), (693, np.float64(402.0)), (694, np.float64(313.0)), (695, np.float64(200.0)), (708, np.float64(238.0)), (709, np.float64(280.0)), (710, np.float64(255.0)), (711, np.float64(668.0)), (712, np.float64(838.0)), (713, np.float64(787.0)), (714, np.float64(679.0)), (715, np.float64(525.0)), (716, np.float64(492.0)), (717, np.float64(406.0)), (718, np.float64(305.0)), (719, np.float64(254.0)), (732, np.float64(287.0)), (733, np.float64(309.0)), (734, np.float64(384.0)), (735, np.float64(832.0)), (736, np.float64(824.0)), (737, np.float64(740.0)), (738, np.float64(619.0)), (739, np.float64(468.0)), (740, np.float64(383.0)), (741, np.float64(352.0)), (742, np.float64(289.0)), (743, np.float64(256.0)), (756, np.float64(280.0)), (757, np.float64(254.0)), (758, np.float64(576.0)), (759, np.float64(854.0)), (760, np.float64(791.0)), (761, np.float64(699.0)), (762, np.float64(561.0)), (763, np.float64(405.0)), (764, np.float64(364.0)), (765, np.float64(320.0)), (766, np.float64(290.0)), (767, np.float64(303.0)), (780, np.float64(314.0)), (781, np.float64(289.0)), (782, np.float64(797.0)), (783, np.float64(831.0)), (784, np.float64(752.0)), (785, np.float64(640.0)), (786, np.float64(516.0)), (787, np.float64(410.0)), (788, np.float64(405.0)), (789, np.float64(393.0)), (790, np.float64(352.0)), (791, np.float64(345.0)), (804, np.float64(276.0)), (805, np.float64(427.0)), (806, np.float64(908.0)), (807, np.float64(810.0)), (808, np.float64(730.0)), (809, np.float64(581.0)), (810, np.float64(497.0)), (811, np.float64(397.0)), (812, np.float64(361.0)), (813, np.float64(404.0)), (814, np.float64(389.0)), (815, np.float64(392.0)), (828, np.float64(259.0)), (829, np.float64(619.0)), (830, np.float64(887.0)), (831, np.float64(780.0)), (832, np.float64(709.0)), (833, np.float64(588.0)), (834, np.float64(452.0)), (835, np.float64(376.0)), (836, np.float64(331.0)), (837, np.float64(409.0)), (838, np.float64(390.0)), (839, np.float64(417.0)), (852, np.float64(292.0)), (853, np.float64(694.0)), (854, np.float64(805.0)), (855, np.float64(683.0)), (856, np.float64(634.0)), (857, np.float64(507.0)), (858, np.float64(451.0)), (859, np.float64(385.0)), (860, np.float64(301.0)), (861, np.float64(311.0)), (862, np.float64(334.0)), (863, np.float64(354.0))]
start_time = time.time()
set_A, set_B, min_diff = minimize_difference(elements)
end_time = time.time()
# compute the sum of set A and set B
print("Set A: {}, Set B: {}, Min diff: {}".format(sum([value for _, value in set_A]), sum([value for _, value in set_B]), min_diff))
print("Time:", end_time - start_time)