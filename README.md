# Clustering Algorithms Visualization

This repository demonstrates various clustering algorithms using Python and visualizes the results using matplotlib. The algorithms included are:

- **K-means++ (Euclidean)**
- **K-means (A*)**
- **DBSCAN**

Each algorithm's clustering result is shown below. The images demonstrate how each algorithm handles data points and obstacles differently. You can run the code to see these visualizations in action.


## Results
### didn't consider obstacle
<div style="display: flex; justify-content: space-between;">
  
  <div style="text-align: center; margin: 10px;">
    <h3>K-means++ (Euclidean)</h3>
    <img src="result/k_means_plus_plus.png" alt="K-means Euclidean" height= "300" width="300">
  </div>


  <div style="text-align: center; margin: 10px;">
    <h3>DBSCAN</h3>
    <img src="result/dbscan.png" alt="DBSCAN" height= "300" width="300">
  </div>

</div>


### consider obstacle


<div style="display: flex; justify-content: space-between;">
  
  <div style="text-align: center; margin: 10px;">
    <h3>K-means A*</h3>
    <img src="result/a_star_k_means.png" alt="K-means Manhattan" height= "300" width="300">
     <p>This algorithm takes more time</p>
  </div>

</div>