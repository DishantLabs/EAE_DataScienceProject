# Improving Snow Cover Classification in Cloudy Mountain Regions Using Machine Learning

Authors: Dishant Sharma, Brock Medernach

## I. Introduction and Background

The Pacific Northwest (PNW) of North America is characterized by a maritime climate strongly influenced by the Pacific Ocean, resulting in high annual precipitation, persistent cloud cover, and significant seasonal snowfall. The Cascade Range, extending from northern California to British Columbia, plays a critical role in shaping this climate through orographic lifting, where moist air masses are forced upward over mountainous terrain, cooling and condensing to produce clouds and precipitation (Cascade Backcountry Alliance, 2023). This process leads to extensive snow accumulation at higher elevations, making the region a key contributor to downstream water resources.

Snowpack in the Cascade Range serves as a natural reservoir, storing water during the winter and releasing it gradually through spring and summer melt. This process is especially important for urban areas such as Seattle, which rely heavily on snowmelt-driven runoff for municipal water supply (City of Seattle, 2024). Accurate snow cover mapping is therefore essential for hydrologic modeling, water resource management, flood forecasting, and understanding broader climate variability. Errors in snow cover estimation can propagate into significant uncertainties in water availability predictions, particularly in regions where snow dominates seasonal hydrology.

Despite its importance, accurately distinguishing snow from cloud cover using traditional remote sensing techniques remains a persistent challenge. Both snow and clouds exhibit high reflectance in visible wavelengths, particularly in the red, green, and blue (RGB) bands, making them spectrally similar in standard satellite imagery. Although indices such as the Normalized Difference Snow Index (NDSI) are commonly used to differentiate snow from other surfaces, their performance is often limited under conditions of thin cloud cover, mixed pixels, or complex terrain (Hall et al., 2002; Dozier et al., 2009). These limitations are especially pronounced in mountainous regions like the Cascades, where topographic shading, vegetation cover, and frequent cloud presence further complicate classification.
To address these challenges, machine learning (ML) approaches have increasingly been applied to snow cover mapping. Unlike traditional threshold-based methods, ML models can learn complex, nonlinear relationships between spectral features and land cover classes, enabling improved discrimination between snow, clouds, and background surfaces. Algorithms such as Random Forest and gradient boosting methods (e.g., XGBoost) have demonstrated strong performance in remote sensing classification tasks due to their robustness to noise and ability to handle high-dimensional datasets (Belgiu and Drăguț, 2016; Maxwell et al., 2018). These models are particularly effective when trained on well-labeled datasets that incorporate both spectral bands and derived indices.

More recently, deep learning (DL) approaches, particularly convolutional neural networks (CNNs) such as U-Net architectures, have shown promise in further improving classification accuracy by incorporating spatial context in addition to spectral information. These models are capable of capturing spatial patterns and textures within imagery, which can be critical for distinguishing between clouds and snow in complex environments (Zhu et al., 2017; Ma et al., 2019). While DL methods require larger datasets and greater computational resources, they represent a powerful extension of traditional ML approaches for remote sensing applications.

This study focuses on the Seattle watershed in Washington State, where reliable snow cover estimation is essential for understanding regional water availability. Using Landsat satellite imagery from 2020 to 2025, along with derived spectral indices such as NDVI and NDSI, we develop and evaluate a supervised machine learning framework to classify surface conditions into three categories: snow, cloud, and background land cover. By explicitly treating clouds as a separate class rather than masking them, this approach aims to produce a more realistic and operationally useful classification product. The primary objective of this study is to assess the effectiveness of machine learning methods in accurately distinguishing snow from cloud cover under the challenging environmental conditions of the Cascade Range.


## II. Data and Methods

This project focuses on the Cascade Mountain Range in the Pacific Northwest (Washington State and part of British Columbia), a region known for persistent winter cloud cover and complex terrain that makes snow mapping particularly challenging. The study area is limited to a manageable number of sub-regions in order to keep computation realistic while still capturing a range of elevation zones, vegetation conditions, and snow climates. The primary remote sensing dataset Landsat surface reflectance imagery, because it provides high spatial resolution (30 m) and includes spectral bands that are highly sensitive to snow and cloud reflectance. This dataset is available for past few decades, but we have mainly focused on years following 2020 (Table 1).

Five spectral bands are used in this project- Blue, Green, Red, Near InfraRed (NIR), and Short-Wave Infrared (SWIR). Spectral indices such as the Normalized Difference Snow Index (NDSI) and Normalized Difference Vegetation Index (NDVI) are calculated from Landsat bands and used both as predictive features and as a baseline snow-mapping method for comparison. 

The project frames snow mapping as a three-class classification problem consisting of snow, cloud and other. The “other” class consists of land cover such as trees, water, bareground, agricultural land, and built-up areas. Rather than attempting to infer snow under clouds using purely optical data, cloud cover is treated explicitly as its own class to create a more honest and operationally realistic product. 

To create training, validation, and testing labels, all five spectral bands were first loaded into ArcGIS Pro and projected to WGS 1984 UTM Zone 10N. Using the Composite Bands tool in the geoprocessing toolbox, the bands were stacked into a single multiband raster for each year, creating a permanent dataset suitable for modeling (while temporary composites for digitization can alternatively be generated through Imagery → Raster Functions → Data Management → Composite Bands). For each year, a polygon feature class was created with a class label field, and training polygons were digitized for each land cover class. Stratified random sampling was then performed using the “Create Spatial Sampling Locations” tool, ensuring equal representation across classes with 4,000 samples per class and a minimum spacing of 30 meters between points. A class label field was populated using conditional logic (“Cloud” if CID = 1, “Other” if CID = 2, and “Snow” if CID = 3). Finally, the Sample tool from Image Analyst was used to extract spectral values at the sampled locations, producing a table for subsequent analysis and modeling. Model Builder in ArcGIS Pro is used to develop a model to automate sampling process (fig.1). More than 70,000 pixel data points are extracted into csv files where each row represents a single pixel value and columns represent latitude, longitude, band values and indices.

![ArcGIS Model](model.PNG)
> Figure 1. Model created in ArcGIS Pro which takes 5 raster images and a polygon file for three classes, and outputs a labelled class table.


The csv files thus generated are further processed using Pandas Python library in Google Colab environment, which involved data cleaning where null values are removed. To avoid overly optimistic results caused by spatial autocorrelation, the dataset is split using a spatial holdout approach, where training and testing are conducted on separate geographic regions rather than randomly mixing pixels from the same scene. To further avoid data leakage, scenes from different years will be used for training and testing. This ensures that the models are evaluated on their ability to generalize to new areas of the Cascades, which is critical for real-world applicability. To achieve this purpose, four images from 2020 to 2023 are used to create the training set, while the 2024 image is used for validation and 2025 image is used for testing the machine learning model. It is done to ensure there is no temporal data leakage into the model. In addition, model is trained, validated and tested on different Landsat scenes centered on Vancouver, Yakima, and Seattle to minimize data leakage through spatial autocorrelation and improving model’s generalizability to different spatial regions. 

Using Scikit-Learn library, data is standardized and made suitable for ML modelling, A Random Forest classifier is trained using pixel-based feature vectors that include spectral reflectance bands, snow and vegetation indices, and coordinates. Random Forest serves as a strong baseline because it performs well on structured tabular data and is relatively robust to noise. Second, an XGBoost model will be trained using the same feature set. XGBoost is expected to improve performance through gradient boosting and is commonly used in industry due to its high accuracy and efficient handling of nonlinear relationships. Model performance is evaluated using metrics appropriate for segmentation and multi-class classification, including overall accuracy, precision, recall, and F1-score. 
All preprocessing, modeling, and evaluation steps will be implemented in Python using standard geospatial and machine learning libraries. The final output will be designed as a reproducible workflow that produces snow/no-snow/cloud classification maps in georeferenced raster format, along with summary statistics such as snow-covered area by elevation band which could be exported as a csv file. 


## III. Initial Results

After preprocessing and removal of null values, the dataset consisted of 47,783 training samples, 12,000 validation samples, and 11,999 test samples. Predictor variables included Landsat spectral bands (Blue, Green, Red, NIR, and SWIR), derived indices (NDVI and NDSI), and spatial coordinates (X, Y).

The baseline Random Forest classifier achieved a validation accuracy of 0.9946, with precision, recall, and F1-scores ranging from 0.99 to 1.00 across all classes. This level of performance is consistent with previous studies demonstrating that Random Forest performs strongly in remote sensing classification due to its ability to capture nonlinear relationships and handle complex, high-dimensional data (Belgiu and Drăguţ 2016). Similar findings are reported by Maxwell et al. (2018), who show that Random Forest consistently performs well in land-cover classification tasks across diverse environments.

After hyperparameter tuning, the model achieved a slightly improved validation accuracy of 0.9953, with a cross-validation score of 0.9865. The relatively small improvement suggests that model performance is primarily driven by the quality of input features rather than parameter optimization alone. This behavior has also been observed by Millard and Richardson (2015), who emphasize the importance of training data quality in Random Forest classification performance.

The confusion matrix indicates that classification errors are minimal across all classes. The Cloud class showed minor confusion with Snow and Other categories, while the Snow class had only a small number of misclassified pixels. This pattern is expected due to the well-documented spectral similarity between snow and cloud reflectance in visible wavelengths. Hall et al. (2002) demonstrate that both snow and clouds exhibit high reflectance in visible bands, which makes them difficult to distinguish using traditional optical methods. Similarly, Stillinger et al. (2019) identify cloud contamination as a major source of error in snow-cover mapping, particularly in mountainous terrain.
Feature importance analysis reveals that the Normalized Difference Snow Index (NDSI) is the most influential predictor, followed by the Blue and Green spectral bands. This result aligns with previous research showing that NDSI is a highly effective indicator for snow detection because it leverages the contrast between visible and shortwave-infrared reflectance (Dietz et al. 2012). Liu et al. (2020) further demonstrate that spectral indices such as NDSI, combined with SWIR information, significantly improve snow classification performance in mountainous environments.

The overall high classification accuracy observed in this study is consistent with the growing body of literature showing that machine learning approaches outperform traditional threshold-based methods in snow mapping. Luo et al. (2022) show that machine learning models improve classification accuracy in complex terrain and forested regions, while Jin et al. (2022) demonstrate that machine learning methods reduce confusion between snow and cloud classes compared to conventional approaches. Additionally, Zhu et al. (2017) highlight that advanced learning-based methods are particularly effective in extracting meaningful patterns from remote sensing data where traditional approaches fail.

Despite these strong results, the near-perfect validation accuracy should be interpreted with caution. Previous studies have shown that spatial autocorrelation and similarities between training and validation datasets can lead to inflated performance estimates. Karasiak et al. (2022) demonstrate that improper spatial separation can artificially increase classification accuracy in remote sensing applications, while Ploton et al. (2020) show that models evaluated without strict spatial independence often perform poorly when applied to new regions.
Overall, the results indicate that the Random Forest model is highly effective for distinguishing snow, cloud, and background land cover in the Cascade Range. The combination of spectral bands and derived indices provides strong predictive capability, and model performance is consistent with findings from previous machine learning-based snow mapping studies. However, final conclusions regarding model robustness will depend on evaluation using a fully independent test dataset.

In addition to the yearly and diurnal distributions, we examine the distributions of additional features associated with EMLs in the dataset (Fig. 3). Consistent with the literature, EMLs are most frequent in the southern half of the Great Plains in spring, roughly south of 40° N latitude. Vertical profiles associated with EMLs have steep lapse rates, relatively low relative humidity, and sufficient vertical wind shear to support deep, moist convection. Due to the presence of the EML’s capping inversion, many EMLs also have moderate to large MUCIN and fairly high 700 mb temperatures. 


## IV. Summary

This project explores the utility of supervised Machine Learning techniques to improve snow cover classification in the mountain regions which receive high winter precipitation and cloud cover remains high. Traditional remote sensing supervised classification algorithms such maximum likelihood often confuses cloud cover with snow cover leading to overestimation of the latter. We have chosen Pacific-Northwest as the study area based on its climatic regime. We have created an API from automatic satellite data acquisition from Microsoft Planetary Computers, which only requires location coordinates, time period, and cloud cover percentage from user’s end. Six satellite images from Landsat-7, 9 and 9 sensors acquired between 2020 and 2025 are used in this project to create training, validation, and testing datasets for this project. Geoprocessing tools in ArcGIS Pro are used for creating the labelled dataset. Further processing and modelling are done using Python libraries in Google Colab environment. Pandas is used read csv files and to remove null values. Additional features (NDSI and NDVI) are also created from existing features using Pandas. Scikit-Learn is used to standardize the datasets. Random Forest model is trained on 2020-2023 data, validated on 2024 data, and tested on 2025 data using the best performing hyperparameters. Overall, the Random Forest model achieved near-perfect classification accuracy, demonstrating that combining spectral bands with derived indices such as NDSI is highly effective for distinguishing snow, cloud, and background land cover in the Cascade Range. The strong performance across all metrics indicates that the model is robust and capable of generalizing unseen validation data well. 

Next step is to run a XGBoost model which will not take much effort because the data is already prepared.

In future, deep learning models, specifically Convolutions Neural Networks (U-Nets) could be trained to improve the classification. Including additional features such as, elevation, slope, aspect, thermal bands can possibly improve the model performance. Training the model on similar mountain regions from Southern Hemisphere, such as, Patagonian Andes, can possibly introduce more variation that ML and DL models can learn from. 


## V. References

Belgiu, M., and L. Drăguţ, 2016: Random forest in remote sensing: A review of applications and future directions. ISPRS J. Photogramm. Remote Sens., 114, 24–31.

Cascade Backcountry Alliance, 2023: Washington weather: Introduction and resources. [Available online at https://www.cascadebackcountryalliance.com/post/washington-weather-introduction-and-resources.]

City of Seattle, 2024: Seattle Public Utilities. [Available online at https://www.seattle.gov/utilities.]

Dietz, A. J., and Coauthors, 2012: Remote sensing of snow—A review of available methods. Int. J. Remote Sens., 33, 4094–4134.

Dong, C., and Coauthors, 2022: Mapping snow cover in forests using optical remote sensing, machine learning, and time-lapse photography. Remote Sens. Environ.

Hall, D. K., and Coauthors, 2002: MODIS snow-cover products. Remote Sens. Environ., 83, 181–194.

Jin, D., and Coauthors, 2022: An improvement of snow/cloud discrimination from machine learning using geostationary satellite data. Big Earth Data, 6, 739–755.

Karasiak, N., and Coauthors, 2022: Spatial dependence between training and test sets: Another pitfall of classification accuracy assessment in remote sensing. Mach. Learn., 111, 2715–2740.

Liu, C., and Coauthors, 2020: MODIS fractional snow cover mapping using machine learning technology in a mountainous area. Remote Sens., 12, 962.

Luo, J., and Coauthors, 2022: Mapping snow cover in forests using optical remote sensing, machine learning, and time-lapse photography. Remote Sens. Environ., 276.

Ma, L., and Coauthors, 2019: Deep learning in remote sensing applications: A meta-analysis and review. ISPRS J. Photogramm. Remote Sens., 152, 166–177.
Maxwell, A. E., and Coauthors, 2018: Random forest classification in remote sensing: A review of applications and future directions. Remote Sens., 10, 463.

Millard, K., and M. Richardson, 2015: On the importance of training data sample selection in random forest image classification. Remote Sens., 7, 8489–8515.

Ploton, P., and Coauthors, 2020: Spatial validation reveals poor predictive performance of large-scale ecological mapping models. Nat. Commun., 11, 4546.

Richiardi, C., and Coauthors, 2023: Snow cover mapping using random forest classification in alpine regions. Remote Sens., 15, 343.

Stillinger, T., and Coauthors, 2019: Cloud masking for Landsat 8 and MODIS Terra over snow-covered terrain: Error analysis and implications for snow mapping. Water Resour. Res., 55, 10607–10623.

Zhu, X. X., and Coauthors, 2017: Deep learning in remote sensing: A comprehensive review and list of resources. IEEE Geosci. Remote Sens. Mag., 5, 8–36.
