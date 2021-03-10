<template>
    <div class="project">
        <a :href="project.url" target="_blank" rel="noopener noreferrer">
            <p class="project-tag" v-for="tag in project.tags" :key="tag">
               {{tag}}
            </p>
            <h3>
                {{ project.title }}
            </h3>
            <p>
                {{ project.about }}
            </p>
        </a>
    </div>
</template>

<style>
.project {
    width: 100%;
    border-top: 2px solid #005397;
    border-bottom: 2px solid #005397;
    color: #005397;
    font-size: 1.25rem;
}

.project-tag {
    color: white;
    border-radius: 10px;
    background-color:#20AD65;
    display: inline-block;
    padding: 0.25rem 0.25rem 0.25rem 0.25rem;
    margin: 1rem 0.5rem 0rem 0rem;
}

.project h3 {
    font-size: 2rem;
}

</style>

<script>
export default {
    name: "Componentproject",
    props: {
        project: {
            type: Object,
            required: true
        }
    },
}
</script>