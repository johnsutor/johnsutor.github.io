<template>
    <nav>
        <h1>
            John Sutor
        </h1>
        <span class="links">
            <nuxt-link to="about">
                About
            </nuxt-link>
            <a href="">
                Resume
            </a>
        </span>
    </nav>
</template>

<style>
    nav {
        color: #005397;
        display: flex;
        justify-content: space-around;
        align-items: center;
        height: 4rem;
        width: 100%;
        border-bottom: 2px solid  #20AD65;
        font-size: 1.5rem;
    }

    nav .links a{
        margin-left: 1rem;
    }

    a, a:hover, a:focus, a:active {
        text-decoration: none;
        color: inherit;
    }
</style>