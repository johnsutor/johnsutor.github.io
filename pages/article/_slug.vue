<template>
    <article>
        <h2>{{ article.title }}</h2>
        <p> Posted {{ formatDate(article.createdAt)}}, updated {{ formatDate(article.updatedAt) }} </p>
        <nuxt-content :document="article" />
    </article>
</template>

<style>
  article {
    color: #005397;
  }

  @media only screen and (min-width: 768px) {
    article {
      margin-top: 4rem;
      width: 50%;
      display: block;
      margin-left: auto;
      margin-right: auto;
    }
  }


  @media only screen and (max-width: 768px) {
    article {
      width: 80vw;
      margin-top: 2rem;
      margin-left: auto;
      margin-right: auto;
    }
  }

  .nuxt-content img {
    width: 100%;
  }

  .nuxt-content a {
    font-weight: bold;
  }
</style>

<script>
  export default {
    async asyncData({ $content, params }) {
      const article = await $content('articles/', params.slug).sortBy('createdAt', 'desc').fetch()

      return { article }
    },
    methods: {
        formatDate(date) {
        const options = { year: 'numeric', month: 'long', day: 'numeric' }
        return new Date(date).toLocaleDateString('en', options)
        }
    }
  }

</script>