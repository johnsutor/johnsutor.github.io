<template>
    <article>
        <h2>{{ article.title }}</h2>
        <p> Posted {{ formatDate(article.createdAt)}} </p>
        <nuxt-content :document="article" />
    </article>
</template>

<style scoped>
  article {
    margin-top: 4rem;
    width: 50%;
    display: block;
    margin-left: auto;
    margin-right: auto;
    color: #005397;
  }
</style>

<script>
  export default {
    async asyncData({ $content, params }) {
      const article = await $content('articles', params.slug).fetch()

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