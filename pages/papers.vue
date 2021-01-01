<template>
    <div class="container">
      <Paper v-for="(paper, index) in papers" 
        :key="index"
        :paper="paper">
      </Paper>
    </div>
</template>

<style scoped>
  .container {
    margin-top: 4rem;
    width: 50%;
    display: block;
    margin-left: auto;
    margin-right: auto;
    color: #005397;
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