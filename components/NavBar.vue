<template>
    <nav>
        <h1>
            <nuxt-link to="/">
                John Sutor
            </nuxt-link>
        </h1>
        <span class="links">
            <nuxt-link to="papers">
                Papers
            </nuxt-link>
            <a 
                href="https://drive.google.com/file/d/1w88hgr2OoN7RMb5bmTbimP6cLRHBevsI/view?usp=sharing"
                target="_blank"
            >
                Resume
            </a>
        </span>
    </nav>
</template>

<style>
    @media only screen and (min-width: 768px) {
        nav {
            font-size: 1.5rem;
        }
    }

    nav {
        color: #005397;
        display: flex;
        justify-content: space-around;
        align-items: center;
        height: 4rem;
        width: 100%;
        border-bottom: 2px solid  #20AD65;
    }

    nav .links a{
        margin-left: 1rem;
    }

    a, a:hover, a:focus, a:active {
        text-decoration: none;
        color: inherit;
    }
</style>