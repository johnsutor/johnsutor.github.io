<template>
    <div class="paper">
        <a :href="paper.url" target="_blank" rel="noopener noreferrer">
            <p class="paper-date">
                {{ paper.conference }}
            </p>
            <h3>
                {{ paper.title }}
            </h3>
            <p>
                {{ paper.abstract }}
            </p>
        </a>
    </div>
</template>

<style>
.paper {
    width: 100%;
    border-top: 2px solid #005397;
    border-bottom: 2px solid #005397;
    color: #005397;
    font-size: 1.25rem;
}

.paper-date {
    color: #20AD65;
}

.paper h3 {
    font-size: 2rem;
}

</style>

<script>
export default {
    name: "ComponentPaper",
    props: {
        paper: {
            type: Object,
            required: true
        }
    },
}
</script>