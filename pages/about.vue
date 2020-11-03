<template>
  <div class="container">
    <h1>
      About Me
    </h1>
    <p>
      Howdy! I'm a current Junior at Florida State University majoring in
      Computational Science and Applied Mathematics. I conduct research under
      Professor Jonathan Adams on the topic of Computer Vision and Synthetic
      Data. I'm also one of the co-founders of
      <a href="https://sciteens.org" target="_blank">
        SciTeens
      </a>
      . We aim to bring scientific research, especially data-centric scientific
      research, to all students regardless of their background. When I'm not
      nerding out, my hobbies include meal prepping, camping/hiking, and vinyl
      collecting.
    </p>
  </div>
</template>

<style>
.container {
  color: #005397;
}

a {
  font-weight: bold;
}
@media only screen and (max-width: 768px) {
  .container {
    width: 80vw;
    margin-top: 2rem;
    margin-left: auto;
    margin-right: auto;
  }
}

@media only screen and (min-width: 768px) {
  .container {
    margin: 4rem 8rem 4rem 8rem;
  }
}
</style>
