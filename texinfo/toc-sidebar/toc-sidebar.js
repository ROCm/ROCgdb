/* Copyright (C) 2026 Free Software Foundation, Inc.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* Build the left navigation sidebar of the HTML manual.

   Each page ships with an empty sidebar shell (put there by
   toc-sidebar.init) and pulls in toc-sidebar-data.js, which sets
   window.TOC_SIDEBAR_CONTENTS to the manual's whole table of contents
   as a string of HTML.  This script turns that string into the
   sidebar tree, marks the current entry for the current page, and
   makes interior nodes collapsible.  The sections the user expands
   stay open when moving from one page to the next.

   With JavaScript off, this script never runs -- the stylesheet hides
   the sidebar and the page shows a single content column.
*/

(function ()
{
  "use strict";

  /* Remember which sections the user has expanded.  Each page is a
     separate file load that rebuilds the tree from scratch, so
     without this, a manual expansion would be lost on when navigating
     between pages.

     The set is a map from a node's href to true, kept in
     sessionStorage as JSON.  This relies on the hrefs being the same
     on every page, which is true because every page builds its tree
     from the the same shared toc-sidebar-data.js.  That is what makes
     a set saved on one page apply to the next.  Storage is not always
     available (private browsing, or some browsers under file://), so
     every access is guarded and falls back to an empty set.  */
  var STORAGE_KEY = "toc-sidebar-open";

  /* The set of open sections, or an empty set when none are stored
     yet or storage is unavailable.  */
  function loadOpenSet ()
  {
    try
      {
        var raw = window.sessionStorage.getItem (STORAGE_KEY);
        return raw ? JSON.parse (raw) : {};
      }
    catch (e)
      {
        return {};
      }
  }

  /* Remember the set of open sections.  */
  function saveOpenSet (open)
  {
    try
      {
        window.sessionStorage.setItem (STORAGE_KEY, JSON.stringify (open));
      }
    catch (e)
      {
        /* Persistence is a nice-to-have, so a failure here is
	   fine.  */
      }
  }

  /* How far down the sidebar is scrolled is remembered the same way
     as the open set.  Each page is a separate file whose sidebar
     starts scrolled to the top, so without this the scroll position
     lost on every navigation.  The value is the sidebar's scrollTop
     in pixels, as a string, in sessionStorage.  */
  var SCROLL_KEY = "toc-sidebar-scroll";

  /* The stored sidebar scroll offset in pixels, or null when there is
     none yet or storage is unavailable.  */
  function loadScroll ()
  {
    try
      {
        var raw = window.sessionStorage.getItem (SCROLL_KEY);
        return raw === null ? null : parseFloat (raw);
      }
    catch (e)
      {
        return null;
      }
  }

  /* Remember the sidebar scroll offset, given in pixels.  */
  function saveScroll (px)
  {
    try
      {
        window.sessionStorage.setItem (SCROLL_KEY, px);
      }
    catch (e)
      {
        /* Persistence is a nice-to-have, so a failure here is
	   fine.  */
      }
  }

  /* Return whether LINK's box lies entirely within SIDEBAR's visible
     area, so the user can already see it without the sidebar
     moving.  */
  function entryVisible (sidebar, link)
  {
    var sr = sidebar.getBoundingClientRect ();
    var lr = link.getBoundingClientRect ();
    return lr.top >= sr.top && lr.bottom <= sr.bottom;
  }

  /* Save the SIDEBAR's scroll offset whenever the user scrolls it, so
     that the next page opens at the same place.  */
  function rememberScroll (sidebar)
  {
    /* Defer the write to the next animation frame so a burst of
       scroll events collapses into one write rather than hitting
       storage on every pixel.  */
    var pending = false;
    sidebar.addEventListener ("scroll", function ()
    {
      if (pending)
        return;
      pending = true;
      window.requestAnimationFrame (function ()
      {
        pending = false;
        saveScroll (sidebar.scrollTop);
      });
    });
  }

  /* Basename of the current document.  For a directory URL, returns
     index.html.  */
  function currentFile ()
  {
    var path = window.location.pathname;
    var base = path.substring (path.lastIndexOf ("/") + 1);
    return base === "" ? "index.html" : base;
  }

  /* Basename of a link target, with any #anchor dropped.  */
  function linkFile (href)
  {
    var hash = href.indexOf ("#");
    if (hash >= 0)
      href = href.substring (0, hash);
    return href.substring (href.lastIndexOf ("/") + 1);
  }

  /* The "#anchor" part of a link target, or empty string when there
     is none.  */
  function linkAnchor (href)
  {
    var hash = href.indexOf ("#");
    return hash >= 0 ? href.substring (hash) : "";
  }

  /* Pick the contents entry for the current page.  Several entries can
     point into the same page when a page holds sections that have no
     node of their own, which texinfo emits as anchors within the one
     file.  Among the entries for this page, prefer the one whose anchor
     matches the address bar.  With no anchor there, that is the entry
     for the top of the page itself.  */
  function pickCurrent (candidates)
  {
    var hash = window.location.hash;
    var i;
    for (i = 0; i < candidates.length; i++)
      if (linkAnchor (candidates[i].getAttribute ("href")) === hash)
        return candidates[i];
    for (i = 0; i < candidates.length; i++)
      if (linkAnchor (candidates[i].getAttribute ("href")) === "")
        return candidates[i];
    return candidates.length ? candidates[0] : null;
  }

  /* Highlight the entry for the current page and anchor, and open the
     path of sections down to it.  CANDIDATES are the entries that
     point at the current page; which one is current depends on the
     address-bar anchor, so this clears any previous highlight first
     and reruns on every anchor change, not only on load -- see the
     listener in decorate.  Returns the highlighted entry, or null
     when none matches.  */
  function markCurrent (container, candidates, open)
  {
    var previous = container.querySelector (".toc-sidebar-current");
    if (previous)
      previous.classList.remove ("toc-sidebar-current");

    var current = pickCurrent (candidates);
    if (!current)
      return null;

    current.classList.add ("toc-sidebar-current");

    /* Open the path down to the current entry, and remember it the
       same way a manual expansion is remembered.  Otherwise a section
       opened only because the user walked into it would collapse
       again the moment they moved to another part of the manual.  */
    var node = current.closest ("li");
    while (node)
      {
        node.classList.add ("toc-sidebar-open");
        var nodeLink = node.querySelector (":scope > a");
        if (nodeLink)
          open[nodeLink.getAttribute ("href")] = true;
        node = node.parentElement ? node.parentElement.closest ("li") : null;
      }
    saveOpenSet (open);

    return current;
  }

  /* Bring CURRENT on screen, but only when it is not already visible.
     When the user moves between two nearby entries, the sidebar stays
     put.  Only a jump to a far-off part of the TOC scrolls.  */
  function revealCurrent (sidebar, current)
  {
    if (sidebar && current && !entryVisible (sidebar, current))
      current.scrollIntoView ({ block: "center" });
  }

  /* Make the plain TOC tree interactive.  Flag every node that has
     children (toc-sidebar-has-children) and give it a [+]/[-] toggle.
     Then, find the entry for the current page, highlight it, and open
     the path down to it.  */
  function decorate (container)
  {
    var here = currentFile ();
    var open = loadOpenSet ();
    var candidates = [];
    var sidebar = document.getElementById ("toc-sidebar");

    var items = container.querySelectorAll ("li");
    items.forEach (function (li)
    {
      var sublist = li.querySelector (":scope > ul");
      var link = li.querySelector (":scope > a");
      var key = link ? link.getAttribute ("href") : null;

      if (sublist)
        {
          li.classList.add ("toc-sidebar-has-children");
          if (key && open[key])
            li.classList.add ("toc-sidebar-open");
          var toggle = document.createElement ("span");
          toggle.className = "toc-sidebar-toggle";
          toggle.setAttribute ("role", "button");
          toggle.setAttribute ("aria-label", "Toggle section");
          toggle.addEventListener ("click", function (event)
          {
            event.preventDefault ();
            var nowOpen = li.classList.toggle ("toc-sidebar-open");
            if (key)
              {
                if (nowOpen)
                  open[key] = true;
                else
                  delete open[key];
                saveOpenSet (open);
              }
          });
          li.insertBefore (toggle, li.firstChild);
        }

      if (link && linkFile (link.getAttribute ("href")) === here)
        candidates.push (link);
    });

    var current = markCurrent (container, candidates, open);
    if (current)
      {
        /* Put the sidebar back where the user had it on the previous
           page, now that the path down to the current entry is open
           and the tree has its final height.  Then center the current
           entry only if that restored position does not already show
           it.  Centering updates scrollTop, so the explicit save
           below records the centered position for the next page.
           Restoring it there then leaves the entry on screen, so the
           next page does not scroll either.  */
        if (sidebar)
          {
            var saved = loadScroll ();
            if (saved !== null)
              sidebar.scrollTop = saved;
            revealCurrent (sidebar, current);
          }
      }
    else if (sidebar)
      {
        /* Nothing matched: we're in the contents page itself, whose
           Top node has no entry in its own table of contents, or some
           auxiliary page.  Leave the tree collapsed -- the top-level
           entries are always shown, and any sections the user
           expanded elsewhere are open through the saved set -- and
           just restore the saved scroll position.  */
        var savedTop = loadScroll ();
        if (savedTop !== null)
          sidebar.scrollTop = savedTop;
      }

    if (sidebar)
      {
        saveScroll (sidebar.scrollTop);
        rememberScroll (sidebar);
      }

    /* Several subsections can share one page, emitted as anchors
       within the one file.  Following such a link from elsewhere on
       that same page is an in-document jump -- the browser moves to
       the anchor without reloading.  In that case this script does
       not run again and the highlight would otherwise stay on the
       entry the user came from.  Handle every hash change so the
       sidebar follows the user into these intra-page sections.  */
    window.addEventListener ("hashchange", function ()
    {
      revealCurrent (sidebar, markCurrent (container, candidates, open));
    });
  }

  /* Replace the placeholder in the shell with the decorated tree.  */
  function inject (ul)
  {
    var toc = document.getElementById ("toc-sidebar-tree");
    if (!toc)
      return;
    toc.innerHTML = "";
    toc.appendChild (ul);
    decorate (toc);
  }

  /* Parse window.TOC_SIDEBAR_CONTENTS and return its top-level <ul>,
     or null.

     This is where the script meets texinfo's generated HTML.  The
     contents tree comes from texinfo's format_contents and is a plain
     nested list:

       <ul>
         <li><a href="Node.html">Title</a>
           <ul> ...child entries... </ul></li>
         ...
       </ul>

     This function, and decorate(), rely only on that generic
     <ul>/<li>/<a href> nesting.  texinfo also puts class names on the
     lists, an id on each <a>, and section numbers in the link text,
     but none of that is read here, so the sidebar is unaffected if
     texinfo changes them.  The hrefs are used as-is and matched
     against the page's own file name (see linkFile).  Both sides come
     out of the same texi2any run, so they agree however texinfo turns
     node names into file names.  */
  function tocTree ()
  {
    if (typeof window.TOC_SIDEBAR_CONTENTS !== "string")
      return null;
    var wrap = document.createElement ("div");
    wrap.innerHTML = window.TOC_SIDEBAR_CONTENTS;
    return wrap.querySelector ("ul");
  }

  /* Put the manual's title at the top of the sidebar.  The title is
     written into toc-sidebar-data.js from the manual's own
     texinfo @settitle.  */
  function setTitle ()
  {
    if (typeof window.TOC_SIDEBAR_TITLE !== "string")
      return;
    var link = document.querySelector (".toc-sidebar-title a");
    if (link)
      link.innerHTML = window.TOC_SIDEBAR_TITLE;
  }

  /* Wire up the filter box.  As the user types, keep only the entries
     whose titles contain the query, along with the ancestors that
     lead to them.  An empty box restores the normal tree.  */
  function setupFilter (toc)
  {
    var input = document.getElementById ("toc-sidebar-search-input");
    if (!input)
      return;
    var items = toc.querySelectorAll ("li");

    /* Match the tree against the current query.  */
    function apply ()
    {
      var query = input.value.trim ().toLowerCase ();

      items.forEach (function (li)
      {
        li.classList.remove ("toc-sidebar-match", "toc-sidebar-show");
      });

      if (query === "")
        {
          toc.classList.remove ("toc-sidebar-filtering");
          return;
        }
      toc.classList.add ("toc-sidebar-filtering");

      items.forEach (function (li)
      {
        var link = li.querySelector (":scope > a");
        var text = link ? link.textContent.toLowerCase () : "";
        if (text.indexOf (query) < 0)
          return;
        li.classList.add ("toc-sidebar-match");
        /* Reveal this entry and the whole chain of <li> above it.  */
        var node = li;
        while (node && node !== toc)
          {
            if (node.tagName === "LI")
              node.classList.add ("toc-sidebar-show");
            node = node.parentElement;
          }
      });
    }

    input.addEventListener ("input", apply);
    input.addEventListener ("keydown", function (event)
    {
      if (event.key === "Escape")
        {
          input.value = "";
          apply ();
        }
    });
  }

  /* Let the user drag the divider between the sidebar and the content
     to choose the sidebar width.  The width is written to the
     --toc-sidebar-width custom property that both panes are laid out
     from, and kept in sessionStorage so that every page opens at the
     same width.  The bounds keep the sidebar from collapsing to
     nothing or swallowing the contents page.  */
  var WIDTH_KEY = "toc-sidebar-width";
  var MIN_WIDTH = 12; /* rem  */
  var MAX_WIDTH = 40; /* rem  */

  /* The stored sidebar width as a string of rem, or null when there
     is none yet or storage is unavailable.  */
  function loadWidth ()
  {
    try
      {
        return window.sessionStorage.getItem (WIDTH_KEY);
      }
    catch (e)
      {
        return null;
      }
  }

  /* Remember the sidebar width, given as a number of rem.  */
  function saveWidth (rem)
  {
    try
      {
        window.sessionStorage.setItem (WIDTH_KEY, rem);
      }
    catch (e)
      {
        /* A stored width is a nice-to-have, so a failure here is
	   fine.  */
      }
  }

  /* Pixels in one rem, for turning a mouse position into rem.  */
  function remPx ()
  {
    return parseFloat (getComputedStyle (document.documentElement).fontSize);
  }

  /* Lay both panes out at the given sidebar width, a number of
     rem.  */
  function applyWidth (rem)
  {
    document.documentElement.style.setProperty ("--toc-sidebar-width",
                                                rem + "rem");
  }

  /* Add the divider handle and make it drag the sidebar width.  Apply
     any width left over from an earlier page first, so the layout
     does not change as the user moves through the manual.  */
  function setupResizer ()
  {
    var handle = document.createElement ("div");
    handle.className = "toc-sidebar-resizer";
    handle.setAttribute ("role", "separator");
    handle.setAttribute ("aria-orientation", "vertical");
    document.body.appendChild (handle);

    var width = parseFloat (loadWidth ());
    if (width >= MIN_WIDTH && width <= MAX_WIDTH)
      applyWidth (width);

    var dragging = false;

    /* Start a drag from a press on the handle.  */
    function begin (event)
    {
      event.preventDefault ();
      dragging = true;
      handle.classList.add ("toc-sidebar-dragging");
      document.body.classList.add ("toc-sidebar-resizing");
    }

    /* Resize to follow the mouse pointer, clamped to the allowed
       range.  */
    function moveTo (clientX)
    {
      if (!dragging)
        return;
      width = clientX / remPx ();
      if (width < MIN_WIDTH)
        width = MIN_WIDTH;
      if (width > MAX_WIDTH)
        width = MAX_WIDTH;
      applyWidth (width);
    }

    /* Finish a drag and store the width it settled on.  */
    function end ()
    {
      if (!dragging)
        return;
      dragging = false;
      handle.classList.remove ("toc-sidebar-dragging");
      document.body.classList.remove ("toc-sidebar-resizing");
      saveWidth (width);
    }

    handle.addEventListener ("mousedown", begin);
    document.addEventListener ("mousemove", function (event)
    {
      moveTo (event.clientX);
    });
    document.addEventListener ("mouseup", end);

    /* Handle touch drags.  touchstart and touchmove made
       passive:false so they can cancel the page scroll that would
       otherwise fight the drag.  */
    handle.addEventListener ("touchstart", begin, { passive: false });
    document.addEventListener ("touchmove", function (event)
    {
      if (!dragging)
        return;
      event.preventDefault ();
      moveTo (event.touches[0].clientX);
    }, { passive: false });
    document.addEventListener ("touchend", end);
  }

  /* Build the sidebar now, as the script runs, rather than waiting
     for the DOMContentLoaded event that fires once the whole page has
     been parsed.  toc-sidebar.init loads this script parser-blocking
     from the body, right after the sidebar shell, so the shell
     already exists here and the tree can be built and scrolled into
     place before the page is painted.  Waiting for DOMContentLoaded
     would instead build it after the first paint, and the user
     would briefly see the sidebar appear and jump on every page load.

     Note this removes the sidebar's own flicker, but not every flash:
     each navigation is still a full document load, and the browser
     can flash the page blank in the gap between the old page and the
     new one.  This is most visible on the big index pages, but seen
     now and then on smaller ones too.  */
  setupResizer ();
  setTitle ();
  var ul = tocTree ();
  if (!ul)
    return;
  inject (ul);
  setupFilter (document.getElementById ("toc-sidebar-tree"));
}) ();
