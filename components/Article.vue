<template>
<div class="article">
<nuxt-link  :to="{name: 'article-slug', params: {slug: article.slug }}">
        <p class="article-date">
            {{ formatDate }}
        </p>
        <h3>
            {{ article.title }}
        </h3>
        <p>
            {{ article.description }}
        </p>
    </nuxt-link>
</div>
    
</template>

<style>
.article {
    width: 100%;
    border-top: 2px solid #005397;
    border-bottom: 2px solid #005397;
    color: #005397;
    font-size: 1.5rem;
}

.article-date {
    color: #20AD65;
}

.article h3 {
    font-size: 2rem;
}


</style>

<script>
export default {
    name: "ComponentArticle",
    props: {
        article: {
            type: Object,
            required: true
        }
    },
    computed: {
        formatDate() {
            let date = new Date(this.article.updatedAt);
            return date.toLocaleDateString('en-us', {
                weekday: "long",
                month: "long",
                day: "numeric",
                hour: "numeric",
                minute: "numeric",
            });
        }
    }
}
</script>