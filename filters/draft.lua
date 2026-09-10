local function is_true(value)
  if value == nil then
    return false
  end
  if value == true then
    return true
  end
  local text = pandoc.utils.stringify(value):lower()
  return text == "true" or text == "yes"
end

local function collect_tags(value)
  local tags = {}
  if value == nil then
    return tags
  end

  if type(value) == "table" and value.t == "MetaList" then
    for _, item in ipairs(value) do
      table.insert(tags, pandoc.utils.stringify(item))
    end
  elseif type(value) == "table" then
    for _, item in ipairs(value) do
      table.insert(tags, pandoc.utils.stringify(item))
    end
  else
    table.insert(tags, pandoc.utils.stringify(value))
  end

  return tags
end

function Pandoc(doc)
  if os.getenv("SHOW_DRAFT_CONTENT") == "true" then
    return doc
  end

  if not is_true(doc.meta.draft) then
    return doc
  end

  local blocks = {}
  local tags = collect_tags(doc.meta.tags)

  if #tags > 0 then
    local tag_blocks = {}
    for _, tag in ipairs(tags) do
      table.insert(
        tag_blocks,
        pandoc.Span({ pandoc.Str(tag) }, pandoc.Attr("", { "draft-tag" }))
      )
    end
    table.insert(blocks, pandoc.Div(tag_blocks, pandoc.Attr("", { "draft-tags" })))
  end

  table.insert(blocks, pandoc.Para({ pandoc.Str("(on progress)") }))

  return pandoc.Pandoc({ pandoc.Div(blocks, pandoc.Attr("", { "draft-placeholder" })) }, doc.meta)
end
