import math
from enum import IntEnum

import torch


class Category(IntEnum):
    Others = 0
    Time = 1
    Money = 2
    Data = 3
    Unitlessless = 4
    Science = 5


class SubCategory(IntEnum):
    Others = 0
    Spatial_Length = 1
    Spatial_Area = 2
    Spatial_Volume = 3
    Time_TimeInterval = 4
    Time_Frequency = 5
    Matter_Mass = 6
    Thermodynamics_Pressure = 7
    Ele_Energy = 8
    Ele_Power = 9
    Money = 10
    Data = 11
    Unitless_Quantity = 12
    Unitless_Rank = 13
    Unitless_Score = 14
    Unitless_Ratio = 15
    Unitless_Factor = 16
    Unitless_Angle = 17
    Thermodynamics_Temperature = 18
    Mechanics_Speed = 19


Msr_Type_Weight = math.sqrt(1203) / torch.sqrt_(
    torch.tensor([197, 79, 31, 11, 176, 26, 56, 12, 14, 18, 1203, 21, 1005, 17, 117, 434, 10],
                 dtype=torch.float))

Category2SubCategory = {
    Category.Time: [SubCategory.Time_TimeInterval, SubCategory.Time_Frequency],
    Category.Money: [SubCategory.Money],
    Category.Data: [SubCategory.Data],
    Category.Unitlessless: [SubCategory.Unitless_Quantity, SubCategory.Unitless_Rank, SubCategory.Unitless_Score,
                            SubCategory.Unitless_Ratio, SubCategory.Unitless_Factor, SubCategory.Unitless_Angle],
    Category.Others: [SubCategory.Others],
    Category.Science: [SubCategory.Spatial_Length, SubCategory.Spatial_Area, SubCategory.Spatial_Volume,
                       SubCategory.Matter_Mass, SubCategory.Thermodynamics_Pressure, SubCategory.Ele_Energy,
                       SubCategory.Ele_Power, SubCategory.Thermodynamics_Temperature, SubCategory.Mechanics_Speed],
    -1: [-1]}

SubCategory2Category = {subCategory: category for category in Category2SubCategory.keys() for subCategory in
                        Category2SubCategory[category]}

Unuse_msr_type = [SubCategory.Others]


class SupportQueryFlag(IntEnum):
    """Whether the field in support or query dataset"""
    Pad = 0
    Support = 1
    Query = 2
