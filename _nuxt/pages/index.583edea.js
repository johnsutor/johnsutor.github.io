(window.webpackJsonp=window.webpackJsonp||[]).push([[4],{191:function(t,e,n){var content=n(197);"string"==typeof content&&(content=[[t.i,content,""]]),content.locals&&(t.exports=content.locals);(0,n(37).default)("7601b944",content,!0,{sourceMap:!1})},192:function(t,e,n){"use strict";n.r(e);var r={name:"ComponentArticle",props:{article:{type:Object,required:!0}},computed:{formatDate:function(){return new Date(this.article.updatedAt).toLocaleDateString("en-us",{weekday:"long",month:"long",day:"numeric",hour:"numeric",minute:"numeric"})}}},o=(n(196),n(15)),component=Object(o.a)(r,(function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"article"},[n("nuxt-link",{attrs:{to:{name:"article-slug",params:{slug:t.article.slug}}}},[n("p",{staticClass:"article-date"},[t._v("\r\n            "+t._s(t.formatDate)+"\r\n        ")]),t._v(" "),n("h3",[t._v("\r\n            "+t._s(t.article.title)+"\r\n        ")]),t._v(" "),n("p",[t._v("\r\n            "+t._s(t.article.description)+"\r\n        ")])])],1)}),[],!1,null,null,null);e.default=component.exports},196:function(t,e,n){"use strict";var r=n(191);n.n(r).a},197:function(t,e,n){(e=n(36)(!1)).push([t.i,".article{width:100%;border-top:2px solid #005397;border-bottom:2px solid #005397;color:#005397;font-size:1.5rem}.article-date{color:#20ad65}.article h3{font-size:2rem}",""]),t.exports=e},198:function(t,e,n){var content=n(208);"string"==typeof content&&(content=[[t.i,content,""]]),content.locals&&(t.exports=content.locals);(0,n(37).default)("b71e53a0",content,!0,{sourceMap:!1})},206:function(t,e,n){t.exports=n.p+"img/john.cee399a.jpg"},207:function(t,e,n){"use strict";var r=n(198);n.n(r).a},208:function(t,e,n){(e=n(36)(!1)).push([t.i,".top[data-v-2a74f2d1]{display:flex;align-items:center;justify-content:center}@media only screen and (max-width:768px){.container[data-v-2a74f2d1]{width:80vw;margin-top:2rem;margin-left:auto;margin-right:auto}.top[data-v-2a74f2d1]{flex-wrap:wrap-reverse}.john-photo[data-v-2a74f2d1]{width:90vw}.top h2[data-v-2a74f2d1]{font-size:3rem;color:#005397}}@media only screen and (min-width:768px){.container[data-v-2a74f2d1]{margin:4rem 8rem}.top-info[data-v-2a74f2d1]{margin:4rem}.top h2[data-v-2a74f2d1]{font-size:5rem;color:#005397}.john-photo[data-v-2a74f2d1]{width:30rem}}a[data-v-2a74f2d1]{font-weight:700}.top-links[data-v-2a74f2d1]{display:flex;font-size:2rem;align-items:center}.about[data-v-2a74f2d1]{margin:4rem 0;font-size:1.5rem;color:#005397}.papers-button[data-v-2a74f2d1]{color:#fff;padding:1rem;border-radius:1rem;background:#005397;background:linear-gradient(135deg,#005397,#20ad65)}.about-contact[data-v-2a74f2d1]{margin-left:1rem;background-clip:text;background-size:100%;background-repeat:repeat;background:#005397;background:linear-gradient(135deg,#005397,#20ad65);-webkit-background-clip:text;-webkit-text-fill-color:transparent;-moz-background-clip:text;-moz-text-fill-color:transparent}.articles[data-v-2a74f2d1]{margin:4rem 0}.articles h2[data-v-2a74f2d1]{font-size:3rem;color:#005397}",""]),t.exports=e},220:function(t,e,n){"use strict";n.r(e);var r={data:function(){return{articles:[]}},components:{Article:n(192).default}},o=(n(207),n(15)),component=Object(o.a)(r,(function(){var t=this,e=t.$createElement,r=t._self._c||e;return r("div",{staticClass:"container"},[r("div",{staticClass:"top"},[r("span",{staticClass:"top-info"},[r("h2",[t._v("\n        Hey, I'm John Sutor\n      ")]),t._v(" "),r("span",{staticClass:"top-links"},[r("nuxt-link",{staticClass:"papers-button",attrs:{to:"papers"}},[t._v("\n          Papers\n        ")]),t._v(" "),r("a",{staticClass:"about-contact",attrs:{href:"mailto:john@sciteens.org"}},[t._v("\n          Contact\n        ")])],1)]),t._v(" "),r("img",{staticClass:"john-photo",attrs:{src:n(206),alt:"John Hiking"}})]),t._v(" "),t._m(0),t._v(" "),r("div",{staticClass:"articles"},[r("h2",[t._v("\n      Blogs\n    ")]),t._v(" "),t._l(t.articles,(function(article){return r("Article",{key:article.title,attrs:{article:article}})}))],2)])}),[function(){var t=this.$createElement,e=this._self._c||t;return e("div",{staticClass:"about"},[e("p",[this._v("\n      Howdy! I'm a current Junior at Florida State University majoring in\n      Computational Science and Applied Mathematics. I conduct research under\n      Professor Jonathan Adams on the topic of Computer Vision and Synthetic\n      Data. I'm also one of the co-founders of\n      "),e("a",{attrs:{href:"https://sciteens.org",target:"_blank"}},[this._v("\n        SciTeens\n      ")]),this._v("\n      . We aim to bring scientific research, especially data-centric scientific\n      research, to all students regardless of their background. When I'm not\n      nerding out, my hobbies include meal prepping, camping/hiking, and vinyl\n      collecting.\n    ")])])}],!1,null,"2a74f2d1",null);e.default=component.exports;installComponents(component,{Article:n(192).default})}}]);