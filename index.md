---
title: Synapse
subtitle: Learning Preferential Concepts from Visual Demonstrations
layout: default
order: 1
description: Synapse learns preferential concepts from natural language input and a few visual demonstrations.
keywords: [Synapse, Concept learning, Neuro-symbolic programming, Program Synthesis, Visual Reasoning]
---

<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500&display=swap');
.curly-font {
    font-family: 'Space Grotesk', cursive;
    color: orange;
}
</style>

<!-- Title and Authors -->
<section class="hero">
  <div class="hero-body">
    <div class="container is-max-desktop">
      <div class="columns is-centered">
        <div class="column has-text-centered">
          <h1 class="title is-1 publication-title">Synapse: Learning Preferential Concepts from Visual Demonstrations</h1>
          <div class="is-size-5 publication-authors">
            <span class="author-block">
              Sadanand Modak<sup>1</sup>, </span>
            <span class="author-block">
              Noah Patton<sup>1</sup>, </span>
            <span class="author-block">
              Isil Dillig<sup>1</sup>, </span>
            <span class="author-block">
              Joydeep Biswas<sup>1</sup></span>
          </div>
          <div class="is-size-5 publication-authors">
            <span class="author-block"><sup>1</sup>UT Austin</span>
          </div>
        </div>
        <div class="column has-text-centered buttons are-medium is-centered">
          <div class="publication-links">
            <!-- Paper PDF Link. -->
            <span class="link-block">
              <a href="https://amrl.cs.utexas.edu/synapse"
                class="external-link button is-normal is-rounded is-dark">
                <span class="icon">
                  <i class="fas fa-file-pdf"></i>
                </span>
                <span>Paper</span>
              </a>
            </span>
            <!-- Paper arxiv Link. -->
            <span class="link-block">
              <a href="https://amrl.cs.utexas.edu/synapse"
                class="external-link button is-normal is-rounded is-dark">
                <span class="icon">
                  <i class="ai ai-arxiv"></i>
                </span>
                <span>ArXiv</span>
              </a>
            </span>
            <!-- Code Link. -->
            <span class="link-block">
              <a href="https://github.com/sadanand1120/nspl"
                class="external-link button is-normal is-rounded is-dark">
                <span class="icon">
                  <i class="fab fa-github"></i>
                </span>
                <span>Code</span>
              </a>
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="section">
  <div class="container is-max-desktop">
    <!-- Abstract. -->
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <h2 class="title is-3">Abstract</h2>
        <div class="content has-text-justified">
          <p>
            We address the problem of <i>preference learning</i>, which aims to learn user-specific preferences
            (e.g., <i> "good parking spot" </i>, <i> "convenient drop-off location" </i>) from visual input. Despite
            its similarity to learning <i>factual concepts</i> (e.g., <i>"red cube"</i>), preference learning is a
            fundamentally harder problem due to its subjective nature and the paucity of person-specific training
            data.
          </p>
          <p>
            We address this problem using a
            new framework called <span class="dnerf">Synapse</span>, which is a neuro-symbolic approach designed to
            efficiently learn
            preferential concepts from limited demonstrations. <span class="dnerf">Synapse</span> represents
            preferences as neuro-symbolic
            programs in a domain-specific language (DSL) that operates over images and leverages a novel combination
            of visual parsing, large language models, and program synthesis to learn programs representing individual
            preferences. We evaluate <span class="dnerf">Synapse</span> through extensive experimentation, including a
            user case study focusing
            on mobility-related concepts in mobile robotics and autonomous driving. Our evaluation demonstrates that
            it significantly outperforms existing baselines as well as its own ablations.
          </p>
        </div>
      </div>
    </div>
    <!--/ Abstract. -->
  </div>
</section>

## <span style="color:red">Website heavily under construction...</span>

<footer class="footer">
  <div class="container">
    <div class="content has-text-centered">
      <a class="icon-link" href="https://amrl.cs.utexas.edu/synapse">
        <i class="fas fa-file-pdf"></i>
      </a>
      <a class="icon-link" href="https://github.com/sadanand1120/nspl" class="external-link" disabled>
        <i class="fab fa-github"></i>
      </a>
    </div>
    <div class="columns is-centered">
      <div class="column is-8">
        <div class="content">
          <p>
            This website uses source code from <a href="https://github.com/nerfies/nerfies.github.io"><span
                class="dnerf">Nerfies</span></a> and <a href="https://github.com/ut-amrl/codebotler"><span
                  class="dnerf">CodeBotler</span></a>.
          </p>
        </div>
      </div>
    </div>
  </div>
</footer>
