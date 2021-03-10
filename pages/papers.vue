<template>
    <div class="container">
      <Paper v-for="(paper, index) in papers" 
        :key="index"
        :paper="paper">
      </Paper>
    </div>
</template>

<style scoped>
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
    margin-top: 4rem;
    width: 50%;
    display: block;
    margin-left: auto;
    margin-right: auto;
    color: #005397;
  }
  }

  .article-image {
    width: 100%;
  }
</style>

<script>
  import Paper from '@/components/Paper'
export default {
  async fetch() {
    await this.$content('papers').sortBy('createdAt', 'desc').fetch().then(
      papers => {
        this.papers = papers
      }
    )
  },
  data() {
    return {
      papers: []
    }
  },
  components: {
    Paper
  }
}
</script>