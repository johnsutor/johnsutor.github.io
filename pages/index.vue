<template>
  <div class="container">
    <!-- Top of Page -->
    <div class="top">
      <span class="top-info">
        <h2>
          Hey, I'm John Sutor
        </h2>
        <span class="top-links">
          <nuxt-link 
            to="about"
            class="about-button"
          >
            About
          </nuxt-link>
            <a
              class="about-contact"
              href="mailto:john@sciteens.org"
            >
              Contact
            </a>
        </span>
      </span>

      <img 
        class="john-photo"
        src="~assets/john.jpg" 
        alt="John Hiking"
      >
    </div>

  </div>
</template>

<style scoped>
  .top {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  @media only screen and (max-width: 768px) {
    .container {
      width: 80vw;
      margin-top: 2rem;
      margin-left: auto;
      margin-right: auto;
    }

    .top {
      flex-wrap: wrap-reverse;
    }

    .john-photo {
      width: 90vw;
    }

    .top h2 {
      font-size: 3rem;
      color: #005397;
    }
  }

  @media only screen and (min-width: 768px) {
    .container {
      margin: 4rem 8rem 4rem 8rem;
    }

    .top-info {
      margin: 4rem;
    }

    .top h2 {
      font-size: 5rem;
      color: #005397;
    }

    .john-photo {
      width: 30rem;
    }
  }

  .top-links {
    display: flex;
    font-size: 2rem;
    align-items: center;
  }

  .about-button {
    color: #FFFFFF;
    padding: 1rem;
    border-radius: 1rem;
    background: rgb(0,83,151);
    background: linear-gradient(135deg, rgba(0,83,151,1) 0%, rgba(32,173,101,1) 100%);
  }

  .about-contact {
    margin-left: 1rem;
    background-clip: text;
    background-size: 100%;
    background-repeat: repeat;
    background: rgb(0,83,151);
    background: linear-gradient(135deg, rgba(0,83,151,1) 0%, rgba(32,173,101,1) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; 
    -moz-background-clip: text;
    -moz-text-fill-color: transparent;
  }
</style>