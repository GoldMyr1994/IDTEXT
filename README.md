# TexTID
Text identification with Stroke Width Transform

## Getting Started

### Requirements
- Python 3.6.4 
- OpenCV 3.4.1
- NumPy 1.14.0
- scikit-image 0.13.1
- ScyPy 1.0.0


### Configuration
| key                          | Description                          | Type  |
| -----------------------------|--------------------------------------| ------|
| input                        | input file path                      | str   |
| save                         | enables the saving of documents in the specified folder in output  | bool  |
| output                       | output folder                        | str   |
| dark_on_light                | text is dark on light background     | bool  |
| deskew                       | deskew configuration                 | Object|
| letters                      | letters configuration                | Object|
| words                        | words configuration                  | Object|

#### Deskew Configuration
>This step could fail. If the threshold is too high, too few lines are found that may not correspond to the text, if too many lines are too high in addition to the text

| key           | Description                                                         | Type          |
| --------------|---------------------------------------------------------------------| --------------|
| enable        | enable skew correction                                              | bool          |
| threshold     | threshold used to find peaks in Hough space, null for auto threshold| int or null   |

#### Letters Configuration
| key                    | Description                                                      | Type    |
| -----------------------|------------------------------------------------------------------| --------|
| min_width              | minimum width in pixels                                          | int     |
| min_height             | minimum height in pixels                                         | int     |
| max_width              | maximum width as a percentage of the width of the image          | float   |
| max_height             | maximum height as a percentage of the height of the image        | float   |
| width_height_ratio     | maximum ratio between height and width                           | float   |
| height_width_ratio     | maximum ratio between width and height                           | float   |
| min_diag_mswt_ratio    | minimum ratio between bounding box diagonal and median swt value | float   |
| max_diag_mswt_ratio    | maximum ratio between bounding box diagonal and median swt value | float   |


#### Words Configuration
| key                | Description                                                                      | Type    |
| -------------------|----------------------------------------------------------------------------------| --------|
| thresh_pairs_y     | maximum distance allowed between the baseline of two letters of the same word    | int     |
| thresh_mswt        | maximum distance allowed between the median swt of two letters of the same word  | int     |
| thresh_height      | maximum distance allowed between the height of two letters of the same word      | int     |
| width_scale        | the maximum horizontal distance between two letters of the same word must be less than the width of the largest letter multiplied by width_scale                                                         | float   |
| height_scale        | the maximum vertical distance between two letters of the same word must be less than the hight of the highest letter multiplied by width_scale                                                         | float   |

### How to run
```
python textid.py config.json
```
some examples with the respective configuration files are available in the examples folder


## Authors

* **Mauro Conte** - [GoldMyr1994](https://github.com/GoldMyr1994)

