#Feature file structure

```
{
    "inputs":
    {
        "token_types":[], # shape: col_num
        "segments":[], # shape: col_num 
        "categories":[[]], # shape: col_num, 9 
        "semantic_embeds":[[]], # shape: col_num, 768  
        "data_characters":[[]], # shape: col_num, 33 
        "mask":[], # shape: col_num
        "field_indices":[], # shape: col_num
    },
    "outputs":[[]], # shape: col_num, 16; [Dim_label, Msr_label, Gby_score, Key_score, Msr_score, Msr_type_subcategory, Msr_type_category, Agg_label_start, Agg_label_end] 
}
```