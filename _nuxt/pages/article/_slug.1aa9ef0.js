(window.webpackJsonp=window.webpackJsonp||[]).push([[3],{191:function(t,e,r){var content=r(197);"string"==typeof content&&(content=[[t.i,content,""]]),content.locals&&(t.exports=content.locals);(0,r(37).default)("7601b944",content,!0,{sourceMap:!1})},192:function(t,e,r){"use strict";r.r(e);var n={name:"ComponentArticle",props:{article:{type:Object,required:!0}},computed:{formatDate:function(){return new Date(this.article.updatedAt).toLocaleDateString("en-us",{weekday:"long",month:"long",day:"numeric",hour:"numeric",minute:"numeric"})}}},o=(r(196),r(15)),component=Object(o.a)(n,(function(){var t=this,e=t.$createElement,r=t._self._c||e;return r("div",{staticClass:"article"},[r("nuxt-link",{attrs:{to:{name:"article-slug",params:{slug:t.article.slug}}}},[r("p",{staticClass:"article-date"},[t._v("\r\n            "+t._s(t.formatDate)+"\r\n        ")]),t._v(" "),r("h3",[t._v("\r\n            "+t._s(t.article.title)+"\r\n        ")]),t._v(" "),r("p",[t._v("\r\n            "+t._s(t.article.description)+"\r\n        ")])])],1)}),[],!1,null,null,null);e.default=component.exports},195:function(t,e,r){var content=r(205);"string"==typeof content&&(content=[[t.i,content,""]]),content.locals&&(t.exports=content.locals);(0,r(37).default)("007e9ba0",content,!0,{sourceMap:!1})},196:function(t,e,r){"use strict";var n=r(191);r.n(n).a},197:function(t,e,r){(e=r(36)(!1)).push([t.i,".article{width:100%;border-top:2px solid #005397;border-bottom:2px solid #005397;color:#005397;font-size:1.5rem}.article-date{color:#20ad65}.article h3{font-size:2rem}",""]),t.exports=e},204:function(t,e,r){"use strict";var n=r(195);r.n(n).a},205:function(t,e,r){(e=r(36)(!1)).push([t.i,"article{color:#005397}@media only screen and (min-width:768px){article{margin-top:4rem;width:50%;display:block;margin-left:auto;margin-right:auto}}@media only screen and (max-width:768px){article{width:80vw;margin-top:2rem;margin-left:auto;margin-right:auto}}.article-image{width:100%;height:10rem}.nuxt-content a{font-weight:700}",""]),t.exports=e},218:function(t,e,r){"use strict";r.r(e);r(24);var n=r(2),o={asyncData:function(t){return Object(n.a)(regeneratorRuntime.mark((function e(){var r,n,article;return regeneratorRuntime.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return r=t.$content,n=t.params,e.next=3,r("articles",n.slug).fetch();case 3:return article=e.sent,e.abrupt("return",{article:article});case 5:case"end":return e.stop()}}),e)})))()},methods:{formatDate:function(t){return new Date(t).toLocaleDateString("en",{year:"numeric",month:"long",day:"numeric"})}}},c=(r(204),r(15)),component=Object(c.a)(o,(function(){var t=this,e=t.$createElement,r=t._self._c||e;return r("article",[r("h2",[t._v(t._s(t.article.title))]),t._v(" "),r("p",[t._v(" Posted "+t._s(t.formatDate(t.article.createdAt))+" ")]),t._v(" "),r("nuxt-content",{attrs:{document:t.article}})],1)}),[],!1,null,null,null);e.default=component.exports;installComponents(component,{Article:r(192).default})}}]);