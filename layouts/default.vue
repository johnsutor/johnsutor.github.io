<template>
  <div>
    <NavBar></NavBar>
    <Nuxt />
  </div>
</template>

<style>
  body {
    max-width: 100%;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
  }
</style>

<script>
import NavBar from '@/components/NavBar'
export default {
  components: [
    NavBar
  ]
}
</script>