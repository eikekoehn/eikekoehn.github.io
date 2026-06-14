---
layout: single
classes: wide
title: Scientific visualizations
permalink: /visualizations/
author_profile: true
tagline: " "
#header:
#    #image: /assets/images/north_atlantic_3D_shaded_T.png
#    video: /assets/videos/animation_T.mp4
---

<video autoplay muted loop playsinline controls width="100%">
  <source src="/assets/videos/animation_T.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<div class="grid-container">
  <div class="grid-row">
    <div class="grid-item">
      <a href="https://www.youtube.com/watch?v=cEUvoY8krHM&t=0s" class="card-link" target="_blank">
        <div class="card">
          <img src="/assets/images/gregor_et_al_2024_snapshot.png" alt="Image Gregor2024">
          <div class="card-title">Air-sea CO<sub>2</sub> flux product OceanSODA-ETHZv2 from Gregor et al. (2024)</div>
        </div>
      </a>
    </div>
    <div class="grid-item">
      <a href="https://www.youtube.com/watch?v=_S9IrLUR8RU" class="card-link" target="_blank">
        <div class="card">
          <img src="/assets/images/vendee_globe_snapshot.png" alt="Image VendeeGlobe">
          <div class="card-title">Air-sea CO<sub>2</sub> flux (Gregor et al., 2024) during Vendée Globe 2020/21</div>
        </div>
      </a>
    </div>
    <div class="grid-item">
      <a href="{{ '/assets/videos/animation_small.mov' | relative_url }}" class="card-link" target="_blank">
        <div class="card">
          <img src="/assets/images/omz_in_swm.png" alt="Image OMZ_in_SWM">
          <div class="card-title">An oxygen minimum zone in a shallow water model (Köhn et al., 2024)</div>
        </div>
      </a>
    </div>
  </div>
</div>

<div class="grid-container">
  <div class="grid-row">
    <div class="grid-item">
      <a href="_pages/animations/woa2023.html" class="card-link" target="_blank">
        <div class="card">
          <img src="/assets/images/north_atlantic_3D_shaded_T.png" alt="North Atlantic temperatures">
          <div class="card-title">
            Climatological distributions of ocean properties (World Ocean Atlas 2023)
          </div>
        </div>
      </a>
    </div>
  </div>
</div>


<style>
.grid-container {
    display: flex;
    justify-content: center;
    padding: 20px;
}
.grid-row {
    display: flex;
    gap: 20px;
}
.grid-item {
    flex: 1;
    max-width: 300px;
}
.card {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    text-align: center;
    overflow: hidden;
}
.card img {
    width: 100%;
    height: auto;
}
.card-title {
    padding: 10px;
    font-size: 1.0em;
    font-weight: bold; 
    color: #333;
}
.card-link {
    text-decoration: none; /* Remove underline on the link */
    color: inherit; /* Ensure text color inheritance */
    display: block; /* Makes the entire card clickable */
}
.card-link:hover .card {
    transform: scale(1.05); /* Scale up the card on hover */
}
</style>